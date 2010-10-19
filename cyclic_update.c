/* cyclic_update.c
 *
 * P. Demorest, 2010
 *
 * Functions for implementing Mark's original 'udpate' algorithm
 * for profile and filter separately.
 */
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include "cyclic_utils.h"
#include "cyclic_update.h"

int cyclic_update_profile(struct profile_harm *out, 
        CS *cs_shifted_pos, CS *cs_shifted_neg,
        struct filter_freq *h_shift_array_pos,
        struct filter_freq *h_shift_array_neg) {

    /* Only valid for 1-pol data now */
    if (cs_shifted_pos->npol!=1) { return(-1); }

    /* Loop sums over nu for each alpha */
    int iharm, ichan;
    for (iharm=0; iharm<cs_shifted_pos->nharm; iharm++) {

        fftw_complex top = 0.0;
        double bottom = 0.0;

        for (ichan=0; ichan<cs_shifted_pos->nchan; ichan++) {

            /* Positive alpha */
            fftwf_complex *cs = get_cs(cs_shifted_pos,iharm,0,ichan);
            fftwf_complex h  = h_shift_array_pos[0].data[ichan];
            fftwf_complex ha = h_shift_array_pos[iharm].data[ichan];

            top += (*cs) * conj(h) * ha;
            bottom += creal(h*conj(h)) * creal(ha*conj(ha));

            /* negative alpha */
            cs = get_cs(cs_shifted_neg,iharm,0,ichan);
            h  = h_shift_array_neg[0].data[ichan];
            ha = h_shift_array_neg[iharm].data[ichan];

            top += (*cs) * h * conj(ha);
            bottom += creal(h*conj(h)) * creal(ha*conj(ha));

        }

        //printf("update_profile iharm=%d top=(%+.3e,%+.3e), bottom=%+.3e\n",
        //        iharm, creal(top), cimag(top), bottom);

        out->data[iharm] = top / bottom;
    }

    return(0);
}
int cyclic_update_filter(struct filter_freq *out, 
        CS *cs_shifted_pos,  CS *cs_shifted_neg,
        struct profile_harm *s, 
        struct filter_freq *h_shift_array_pos, 
        struct filter_freq *h_shift_array_neg, 
        int max_harm) {

    /* Only valid for 1-pol data now */
    if (cs_shifted_pos->npol!=1) { return(-1); }
    if (cs_shifted_neg->npol!=1) { return(-1); }

    if (max_harm<=0) max_harm = cs_shifted_pos->nharm;

    /* Loop sums over alpha for each nu */
    int iharm, ichan;
    for (ichan=0; ichan<cs_shifted_pos->nchan; ichan++) {

        fftw_complex top = 0.0;
        double bottom = 0.0;

        /* alpha loop ignores DC */
        for (iharm=1; iharm<max_harm; iharm++) {


            /* positive alpha */
            fftwf_complex sa = s->data[iharm];
            fftwf_complex *cs = get_cs(cs_shifted_pos,iharm,0,ichan);
            fftwf_complex ha = h_shift_array_pos[iharm].data[ichan];

            top += (*cs) * ha * conj(sa);
            bottom += creal(ha*conj(ha)) * creal(sa * conj(sa));

            /* negative alpha */
            /* Profile and data need a extra conjugate here */
            sa = conj(s->data[iharm]);
            cs = get_cs(cs_shifted_neg,iharm,0,ichan);
            ha = h_shift_array_neg[iharm].data[ichan];

            top += conj(*cs) * ha * conj(sa);
            bottom += creal(ha*conj(ha)) * creal(sa * conj(sa));

        }

        //printf("update_filter ichan=%d top=(%+.3e,%+.3e), bottom=%+.3e\n",
        //        ichan, creal(top), cimag(top), bottom);
        
        out->data[ichan] = bottom==0.0 ? 0.0 : top/bottom;

    }

    return(0);

}
