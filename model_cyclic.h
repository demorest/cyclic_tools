/* model_cyclic.h
 *
 * P. Demorest, 2010
 */
#ifndef _MODEL_CYCLIC_H
#define _MODEL_CYCLIC_H

#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include "cyclic_utils.h"

int model_cyclic(CS *out, struct profile_harm *s0, 
        struct filter_time *h, 
        struct filter_freq *h_shift_pos,
        struct filter_freq *h_shift_neg,
        struct cyclic_work *w);

#endif
