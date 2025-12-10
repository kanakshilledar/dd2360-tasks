#ifndef PARTICLES_GPU_H
#define PARTICLES_GPU_H

#include <math.h>

#include "Alloc.h"
#include "Particles.h"
#include "Parameters.h"
#include "PrecisionTypes.h"
#include "Grid.h"
#include "EMfield.h"
#include "InterpDensSpecies.h"

/** particle mover */
int mover_PC_gpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param);

#endif
