#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>

// include project headers so FPpart, FPfield, structs are available
#include "Particles.h"
#include "Alloc.h"
#include "Grid.h"
#include "EMfield.h"
#include "Parameters.h"

// Helper macro for CUDA error check
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> " \
                  << cudaGetErrorString(err) << " (" << err << ")\n"; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// flatten 3D indexing: (ix,iy,iz) with dims Ny,Nz
__device__ inline int dev_idx3d(int ix, int iy, int iz, int Ny, int Nz) {
    return (ix * Ny + iy) * Nz + iz;
}
inline int host_idx3d(int ix, int iy, int iz, int Ny, int Nz) {
    return (ix * Ny + iy) * Nz + iz;
}

/* device kernel: one thread per particle */
__global__ void mover_PC_kernel(
    long nop,
    // particle arrays on device
    FPpart *d_x, FPpart *d_y, FPpart *d_z,
    FPpart *d_u, FPpart *d_v, FPpart *d_w,
    // per-species scalars
    FPpart d_qom,
    FPpart d_dto2,
    FPpart d_dt_sub,
    int d_NiterMover,
    int d_n_sub_cycles,
    // field arrays (flattened)
    const FPfield *d_Ex, const FPfield *d_Ey, const FPfield *d_Ez,
    const FPfield *d_Bxn, const FPfield *d_Byn, const FPfield *d_Bzn,
    // grid node coordinates (flattened)
    const FPpart *d_XN, const FPpart *d_YN, const FPpart *d_ZN,
    // grid scalars
    FPpart xStart, FPpart yStart, FPpart zStart,
    FPpart invdx, FPpart invdy, FPpart invdz,
    FPpart invVOL,
    int Nx, int Ny, int Nz,
    FPpart Lx, FPpart Ly, FPpart Lz,
    // BC flags
    bool PERIODICX, bool PERIODICY, bool PERIODICZ
) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nop) return;

    // temporary variables
    FPpart xptilde, yptilde, zptilde;
    FPpart uptilde = 0.0, vptilde = 0.0, wptilde = 0.0;
    FPfield Exl, Eyl, Ezl, Bxl, Byl, Bzl;

    // local variables copied from global arrays
    FPpart xi[2], eta[2], zeta[2];

    // loop over subcycling steps
    for (int i_sub = 0; i_sub < d_n_sub_cycles; ++i_sub) {
        xptilde = d_x[i];
        yptilde = d_y[i];
        zptilde = d_z[i];

        for (int inner = 0; inner < d_NiterMover; ++inner) {
            // compute cell indices (as in CPU code)
            int ix = 2 + int((d_x[i] - xStart) * invdx);
            int iy = 2 + int((d_y[i] - yStart) * invdy);
            int iz = 2 + int((d_z[i] - zStart) * invdz);

            // compute flattened indices for grid node arrays
            int id_xn_im1 = dev_idx3d(ix - 1, iy, iz, Ny, Nz); // XN[ix-1][iy][iz]
            int id_yn_iyim1 = dev_idx3d(ix, iy - 1, iz, Ny, Nz); // YN[ix][iy-1][iz]
            int id_zn_izim1 = dev_idx3d(ix, iy, iz - 1, Ny, Nz); // ZN[ix][iy][iz-1]
            int id_xn_ixiyiz = dev_idx3d(ix, iy, iz, Ny, Nz);

            // distances from nodes (same formula as CPU)
            xi[0]   = d_x[i] - d_XN[id_xn_im1];
            eta[0]  = d_y[i] - d_YN[id_yn_iyim1];
            zeta[0] = d_z[i] - d_ZN[id_zn_izim1];
            xi[1]   = d_XN[id_xn_ixiyiz] - d_x[i];
            eta[1]  = d_YN[id_xn_ixiyiz] - d_y[i]; // id_xn_ixiyiz==id_yn_ixiyiz since flattened indices consistent
            zeta[1] = d_ZN[id_xn_ixiyiz] - d_z[i];

            // initialize local fields
            Exl = Eyl = Ezl = Bxl = Byl = Bzl = (FPfield)0.0;

            // accumulate contribution from 8 surrounding nodes
            for (int ii = 0; ii < 2; ++ii) {
                for (int jj = 0; jj < 2; ++jj) {
                    for (int kk = 0; kk < 2; ++kk) {
                        FPpart w = xi[ii] * eta[jj] * zeta[kk] * invVOL;
                        int f_ix = ix - ii;
                        int f_iy = iy - jj;
                        int f_iz = iz - kk;
                        int fidx = dev_idx3d(f_ix, f_iy, f_iz, Ny, Nz);
                        Exl += w * d_Ex[fidx];
                        Eyl += w * d_Ey[fidx];
                        Ezl += w * d_Ez[fidx];
                        Bxl += w * d_Bxn[fidx];
                        Byl += w * d_Byn[fidx];
                        Bzl += w * d_Bzn[fidx];
                    }
                }
            }

            // Boris-style push algebra (as in CPU)
            FPpart qomdt2 = d_qom * d_dto2;
            FPpart omdtsq = qomdt2 * qomdt2 * (Bxl*Bxl + Byl*Byl + Bzl*Bzl);
            FPpart denom = (FPpart)1.0 / ((FPpart)1.0 + omdtsq);

            FPpart ut = d_u[i] + qomdt2 * Exl;
            FPpart vt = d_v[i] + qomdt2 * Eyl;
            FPpart wt = d_w[i] + qomdt2 * Ezl;
            FPpart udotb = ut * Bxl + vt * Byl + wt * Bzl;

            uptilde = (ut + qomdt2 * (vt * Bzl - wt * Byl + qomdt2 * udotb * Bxl)) * denom;
            vptilde = (vt + qomdt2 * (wt * Bxl - ut * Bzl + qomdt2 * udotb * Byl)) * denom;
            wptilde = (wt + qomdt2 * (ut * Byl - vt * Bxl + qomdt2 * udotb * Bzl)) * denom;

            // partial position update
            d_x[i] = xptilde + uptilde * (d_dto2);
            d_y[i] = yptilde + vptilde * (d_dto2);
            d_z[i] = zptilde + wptilde * (d_dto2);
        } // inner

        // final updates (as CPU)
        d_u[i] = (FPpart)2.0 * uptilde - d_u[i];
        d_v[i] = (FPpart)2.0 * vptilde - d_v[i];
        d_w[i] = (FPpart)2.0 * wptilde - d_w[i];
        d_x[i] = xptilde + uptilde * d_dt_sub;
        d_y[i] = yptilde + vptilde * d_dt_sub;
        d_z[i] = zptilde + wptilde * d_dt_sub;

        // apply BCs (exactly as CPU)
        // X-direction
        if (d_x[i] > Lx) {
            if (PERIODICX) d_x[i] = d_x[i] - Lx;
            else { d_u[i] = -d_u[i]; d_x[i] = (FPpart)2.0 * Lx - d_x[i]; }
        }
        if (d_x[i] < (FPpart)0.0) {
            if (PERIODICX) d_x[i] = d_x[i] + Lx;
            else { d_u[i] = -d_u[i]; d_x[i] = -d_x[i]; }
        }

        // Y-direction
        if (d_y[i] > Ly) {
            if (PERIODICY) d_y[i] = d_y[i] - Ly;
            else { d_v[i] = -d_v[i]; d_y[i] = (FPpart)2.0 * Ly - d_y[i]; }
        }
        if (d_y[i] < (FPpart)0.0) {
            if (PERIODICY) d_y[i] = d_y[i] + Ly;
            else { d_v[i] = -d_v[i]; d_y[i] = -d_y[i]; }
        }

        // Z-direction
        if (d_z[i] > Lz) {
            if (PERIODICZ) d_z[i] = d_z[i] - Lz;
            else { d_w[i] = -d_w[i]; d_z[i] = (FPpart)2.0 * Lz - d_z[i]; }
        }
        if (d_z[i] < (FPpart)0.0) {
            if (PERIODICZ) d_z[i] = d_z[i] + Lz;
            else { d_w[i] = -d_w[i]; d_z[i] = -d_z[i]; }
        }
    } // subcycles
} // kernel

/* Host helper: flatten grid-field arrays that are currently pointer-of-pointer-of-pointer style into a 1D vector.
   The CPU code uses indexing like: field->Ex[ix][iy][iz] and grd->XN[ix][iy][iz]
   We will iterate over (ix,iy,iz) ranges and pack into row-major ((ix*Ny)+iy)*Nz+iz
*/
static void flatten_fields_and_grid(
    EMfield *field, grid *grd,
    FPfield **out_Ex, FPfield **out_Ey, FPfield **out_Ez,
    FPfield **out_Bxn, FPfield **out_Byn, FPfield **out_Bzn,
    FPpart **out_XN, FPpart **out_YN, FPpart **out_ZN,
    int Nx, int Ny, int Nz)
{
    size_t tot = (size_t)Nx * Ny * Nz;
    // allocate host arrays (flat)
    *out_Ex  = (FPfield*) malloc(tot * sizeof(FPfield));
    *out_Ey  = (FPfield*) malloc(tot * sizeof(FPfield));
    *out_Ez  = (FPfield*) malloc(tot * sizeof(FPfield));
    *out_Bxn = (FPfield*) malloc(tot * sizeof(FPfield));
    *out_Byn = (FPfield*) malloc(tot * sizeof(FPfield));
    *out_Bzn = (FPfield*) malloc(tot * sizeof(FPfield));
    *out_XN  = (FPpart*)  malloc(tot * sizeof(FPpart));
    *out_YN  = (FPpart*)  malloc(tot * sizeof(FPpart));
    *out_ZN  = (FPpart*)  malloc(tot * sizeof(FPpart));

    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            for (int iz = 0; iz < Nz; ++iz) {
                int id = host_idx3d(ix, iy, iz, Ny, Nz);
                // NOTE: CPU code uses e.g. field->Ex[ix][iy][iz]
                (*out_Ex)[id]  = field->Ex[ix][iy][iz];
                (*out_Ey)[id]  = field->Ey[ix][iy][iz];
                (*out_Ez)[id]  = field->Ez[ix][iy][iz];
                (*out_Bxn)[id] = field->Bxn[ix][iy][iz];
                (*out_Byn)[id] = field->Byn[ix][iy][iz];
                (*out_Bzn)[id] = field->Bzn[ix][iy][iz];

                (*out_XN)[id] = grd->XN[ix][iy][iz];
                (*out_YN)[id] = grd->YN[ix][iy][iz];
                (*out_ZN)[id] = grd->ZN[ix][iy][iz];
            }
        }
    }
}

/* Host wrapper: call this from main instead of mover_PC */
int mover_PC_gpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // echo similar print as CPU version
    std::cout << "***  MOVER_GPU with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    const long nop = part->nop;
    if (nop <= 0) return 0;

    // device pointers for particles
    FPpart *d_x = nullptr, *d_y = nullptr, *d_z = nullptr;
    FPpart *d_u = nullptr, *d_v = nullptr, *d_w = nullptr;

    size_t bytes = (size_t)nop * sizeof(FPpart);

    // allocate device arrays for particle components
    CUDA_CHECK(cudaMalloc((void**)&d_x, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_y, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_z, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_u, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_v, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_w, bytes));

    // copy particle arrays host->device
    CUDA_CHECK(cudaMemcpy(d_x, part->x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, part->y, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, part->z, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u, part->u, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, part->v, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, part->w, bytes, cudaMemcpyHostToDevice));

    // Flatten the fields and grid nodes into 1D host arrays
    int Nx = grd->nxn;
    int Ny = grd->nyn;
    int Nz = grd->nzn;

    // host flattened arrays
    FPfield *h_Ex, *h_Ey, *h_Ez, *h_Bxn, *h_Byn, *h_Bzn;
    FPpart  *h_XN, *h_YN, *h_ZN;

    flatten_fields_and_grid(field, grd,
                            &h_Ex, &h_Ey, &h_Ez,
                            &h_Bxn, &h_Byn, &h_Bzn,
                            &h_XN, &h_YN, &h_ZN,
                            Nx, Ny, Nz);

    // allocate device arrays for fields and grid nodes
    size_t tot = (size_t)Nx * Ny * Nz;
    size_t bytes_field = tot * sizeof(FPfield);
    size_t bytes_node  = tot * sizeof(FPpart);

    FPfield *d_Ex=nullptr, *d_Ey=nullptr, *d_Ez=nullptr, *d_Bxn=nullptr, *d_Byn=nullptr, *d_Bzn=nullptr;
    FPpart  *d_XN=nullptr, *d_YN=nullptr, *d_ZN=nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_Ex,  bytes_field));
    CUDA_CHECK(cudaMalloc((void**)&d_Ey,  bytes_field));
    CUDA_CHECK(cudaMalloc((void**)&d_Ez,  bytes_field));
    CUDA_CHECK(cudaMalloc((void**)&d_Bxn, bytes_field));
    CUDA_CHECK(cudaMalloc((void**)&d_Byn, bytes_field));
    CUDA_CHECK(cudaMalloc((void**)&d_Bzn, bytes_field));
    CUDA_CHECK(cudaMalloc((void**)&d_XN,  bytes_node));
    CUDA_CHECK(cudaMalloc((void**)&d_YN,  bytes_node));
    CUDA_CHECK(cudaMalloc((void**)&d_ZN,  bytes_node));

    // copy fields and nodes to device
    CUDA_CHECK(cudaMemcpy(d_Ex,  h_Ex,  bytes_field, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Ey,  h_Ey,  bytes_field, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Ez,  h_Ez,  bytes_field, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bxn, h_Bxn, bytes_field, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Byn, h_Byn, bytes_field, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bzn, h_Bzn, bytes_field, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_XN,  h_XN,  bytes_node,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_YN,  h_YN,  bytes_node,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ZN,  h_ZN,  bytes_node,  cudaMemcpyHostToDevice));

    // prepare kernel launch params
    int TPB = 256;
    int blocks = (int)((nop + TPB - 1) / TPB);

    // kernel parameters (scalars)
    FPpart qom = part->qom;
    FPpart dt_sub_cycling = (FPpart) param->dt / ((double) part->n_sub_cycles);
    FPpart dto2 = (FPpart)0.5 * dt_sub_cycling;
    FPpart dt_sub = dt_sub_cycling;

    // timing (device)
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    // launch kernel
    mover_PC_kernel<<<blocks, TPB>>>(
        nop,
        d_x, d_y, d_z, d_u, d_v, d_w,
        qom, dto2, dt_sub,
        part->NiterMover,
        part->n_sub_cycles,
        d_Ex, d_Ey, d_Ez, d_Bxn, d_Byn, d_Bzn,
        d_XN, d_YN, d_ZN,
        grd->xStart, grd->yStart, grd->zStart,
        grd->invdx, grd->invdy, grd->invdz,
        grd->invVOL,
        Nx, Ny, Nz,
        grd->Lx, grd->Ly, grd->Lz,
        param->PERIODICX, param->PERIODICY, param->PERIODICZ
    );

    // check kernel and synchronize
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "[mover_PC_gpu] kernel elapsed (ms): " << elapsed_ms << std::endl;

    // copy particles back to host
    CUDA_CHECK(cudaMemcpy(part->x, d_x, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(part->y, d_y, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(part->z, d_z, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(part->u, d_u, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(part->v, d_v, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(part->w, d_w, bytes, cudaMemcpyDeviceToHost));

    // free device memory
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_w));

    CUDA_CHECK(cudaFree(d_Ex));
    CUDA_CHECK(cudaFree(d_Ey));
    CUDA_CHECK(cudaFree(d_Ez));
    CUDA_CHECK(cudaFree(d_Bxn));
    CUDA_CHECK(cudaFree(d_Byn));
    CUDA_CHECK(cudaFree(d_Bzn));
    CUDA_CHECK(cudaFree(d_XN));
    CUDA_CHECK(cudaFree(d_YN));
    CUDA_CHECK(cudaFree(d_ZN));

    // free host flattened arrays
    free(h_Ex); free(h_Ey); free(h_Ez);
    free(h_Bxn); free(h_Byn); free(h_Bzn);
    free(h_XN); free(h_YN); free(h_ZN);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
