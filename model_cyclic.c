/* model_cyclic.c
 *
 * P. Demorest, 2010/10
 *
 * Functions for computing a model cyclic spectrum from 
 * a profile and filter (transfer function).
 */
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include "cyclic_utils.h"
#include "model_cyclic.h"

/* Make freq-shifted filter function arrays, given the info provided
 * in the cyclic spectrum struct.  Shifted arrays are:
 * h_shift_pos[i] = H(nu + alpha_i/2)
 * h_shift_neg[i] = H(nu - alpha_i/2)
 * This function allocates space for the shift arrays.
 */
void make_shifted_filters(CS *c, struct filter_time *h, 
        struct filter_freq **h_shift_pos, struct filter_freq **h_shift_neg,
        struct cyclic_work *w) {

    /* Could double-check dimensions here */

    /* TODO check if shift arrays are already allocated? */

    /* Useful dims */
    const int nh = c->nharm;
    const int nc = c->nchan;

    /* Allocate arrays */
    *h_shift_pos = (struct filter_freq *)malloc(sizeof(struct filter_freq)*nh);
    *h_shift_neg = (struct filter_freq *)malloc(sizeof(struct filter_freq)*nh);
    int i;
    for (i=0; i<nh; i++) {
        (*h_shift_pos)[i].nchan = nc;
        (*h_shift_neg)[i].nchan = nc;
        filter_alloc_freq(&(*h_shift_pos)[i]);
        filter_alloc_freq(&(*h_shift_neg)[i]);
    }

    /* Call shift fn */
    filter_shift(*h_shift_pos, h, nh, +1.0*c->ref_freq/(c->bw*1e6)/2.0, w);
    filter_shift(*h_shift_neg, h, nh, -1.0*c->ref_freq/(c->bw*1e6)/2.0, w);

}

/* Construct a model cyclic spectrum given input profile harmonics and
 * time-domain filter function.  Optionally, pre-computed shifted freq-domain
 * filter arrays can be given to avoid recomputing them.  Output cyclic
 * spectrum must be pre-allocated, and informational params should be filled
 * in.  Only deals with 1-pol data now.
 */
int model_cyclic(CS *out, struct profile_harm *s0, 
        struct filter_time *h, 
        struct filter_freq *h_shift_pos,
        struct filter_freq *h_shift_neg,
        struct cyclic_work *w) {

    /* Check dimensions, etc.
     * Could add more checks here.
     */
    if (out->npol!=1) { return(-1); }

    /* Useful stuff */
    const int nh = out->nharm;
    const int nc = out->nchan;

    /* Compute shifted filter arrays if needed.  Should we make this
     * function able to return these arrays?
     */
    int built_filters = 0;
    if (h_shift_pos==NULL || h_shift_neg==NULL) {
        make_shifted_filters(out, h, &h_shift_pos, &h_shift_neg, w);
        built_filters = 1;
    }

    /* Fill in output values */
    int ih, ic;
    for (ih=0; ih<nh; ih++) {
        for (ic=0; ic<nc; ic++) {
            fftwf_complex *val = get_cs(out, ih, 0, ic);
            *val = h_shift_pos[ih].data[ic] * conj(h_shift_neg[ih].data[ic])
                * s0.data[ih];
        }
    }

    /* Free filter arrays if needed */
    if (built_filters) {
        for (ih=0; ih<nh; ih++) {
            filter_free_freq(&h_shift_pos[ih]);
            filter_free_freq(&h_shift_neg[ih]);
        }
        free(h_shift_pos);
        free(h_shift_neg);
    }

    return(0);
}
