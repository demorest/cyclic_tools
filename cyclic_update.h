/* cyclic_update.h
 *
 * P. Demorest, 2010
 */
#ifndef _CYCLIC_UPDATE_H
#define _CYCLIC_UPDATE_H

#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include "cyclic_utils.h"

/* Return updated S_0 (profile FT) given shifted CS, H */
int cyclic_update_profile(struct profile_harm *out, 
        CS *cs_shifted_pos, CS *cs_shifted_neg,
        struct filter_freq *h_shift_array_pos,
        struct filter_freq *h_shift_array_neg);

/* Return updated H (freq-domain filter) given shifted CS, H, S_0 */
int cyclic_update_filter(struct filter_freq *out, 
        CS *cs_shifted_pos, CS *cs_shifted_neg, 
        struct profile_harm *s, 
        struct filter_freq *h_shift_array_pos,
        struct filter_freq *h_shift_array_neg,
        int max_harm);

/* Compute mean squared difference between model and data
 * to test for convergence.
 */
double cyclic_mse(CS *cs_shifted_pos,  CS *cs_shifted_neg,
        struct profile_harm *s, 
        struct filter_freq *h_shift_array_pos, 
        struct filter_freq *h_shift_array_neg, 
        int max_harm);

#endif
