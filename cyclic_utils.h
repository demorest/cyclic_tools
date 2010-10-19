/* cyclic_utils.h
 *
 * Basic structs/functions to make organizing cyclic spectrum 
 * data easier.
 */
#ifndef _CYCLIC_UTILS_H
#define _CYCLIC_UTILS_H

#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <fitsio.h>

/* Conventions:
 *   nlag = nchan, and are arranged as
 *      0,1,..,nlag/2,-(nlag/2-1),..,-1
 *   Spectra always have nchan points at both pos and neg freqs
 *   Harmonic arrays only have nharm positive components, and include DC
 */

/* Periodic spectrum, with axes phase, pol, chan */
struct periodic_spectrum {
    int nphase;
    int npol;
    int nchan;
    int imjd;
    double fmjd;
    double ref_phase;
    double ref_freq;
    double rf;
    double bw;
    float *data;
};

/* Cyclic spectrum */
struct cyclic_spectrum {
    int nharm;
    int npol;
    int nchan;
    int imjd;
    double fmjd;
    double ref_phase;
    double ref_freq;
    double rf;
    double bw;
    fftwf_complex *data;
};

/* Cyclic correlation */
struct cyclic_correlation {
    int nharm;
    int npol;
    int nlag;
    int imjd;
    double fmjd;
    double ref_phase;
    double ref_freq;
    double rf;
    double bw;
    fftwf_complex *data;
};

/* Periodic correlation */
struct periodic_correlation {
    int nphase;
    int npol;
    int nlag;
    int imjd;
    double fmjd;
    double ref_phase;
    double ref_freq;
    double rf;
    double bw;
    fftwf_complex *data;
};

/* Filter functions in time/freq domain */
struct filter_time {
    int nlag;
    fftwf_complex *data;
};
struct filter_freq {
    int nchan;
    fftwf_complex *data;
};

/* Pulse profiles in phase/harmonic domain */
struct profile_phase {
    int nphase;
    float *data;
};
struct profile_harm {
    int nharm;
    fftwf_complex *data;
};

/* Struct to hold working info (fftw plans, etc) */
struct cyclic_work {

    int npol;

    int nchan;
    int nlag;

    int nphase;
    int nharm;

    /* Add to these as needed */
    fftwf_plan ps2cs;
    fftwf_plan cs2cc;
    fftwf_plan cc2cs;

    fftwf_plan time2freq;
    fftwf_plan freq2time;
    fftwf_plan phase2harm;
    fftwf_plan harm2phase;
};

/* Help save lots of characters... */
typedef struct periodic_spectrum PS;
typedef struct cyclic_spectrum CS;
typedef struct cyclic_correlation CC;
typedef struct periodic_correlation PC;

/* Simple get data funcs to avoid indexing problems, no bounds check for now */
static inline float *get_ps(PS *d, int iphase, int ipol, int ichan) {
    return &d->data[iphase + d->nphase*ipol + d->nphase*d->npol*ichan];
}
static inline fftwf_complex *get_cs(CS *d, int iharm, int ipol, int ichan) {
    return &d->data[iharm + d->nharm*ipol + d->nharm*d->npol*ichan];
}
static inline fftwf_complex *get_cc(CC *d, int iharm, int ipol, int ilag) {
    return &d->data[iharm + d->nharm*ipol + d->nharm*d->npol*ilag];
}
static inline fftwf_complex *get_pc(PC *d, int iphase, int ipol, int ilag) {
    return &d->data[iphase + d->nphase*ipol + d->nphase*d->npol*ilag];
}

/* Alloc/free datatypes */
void cyclic_alloc_ps(PS *d);
void cyclic_alloc_cs(CS *d);
void cyclic_alloc_cc(CC *d);
void cyclic_alloc_pc(PC *d);
void cyclic_free_ps(PS *d);
void cyclic_free_cs(CS *d);
void cyclic_free_cc(CC *d);
void cyclic_free_pc(PC *d);
void filter_alloc_time(struct filter_time *f);
void filter_alloc_freq(struct filter_freq *f);
void filter_free_time(struct filter_time *f);
void filter_free_freq(struct filter_freq *f);
void profile_alloc_phase(struct profile_phase *p);
void profile_alloc_harm(struct profile_harm *p);
void profile_free_phase(struct profile_phase *p);
void profile_free_harm(struct profile_harm *p);

/* Load dimension params from datafile */
int cyclic_load_params(fitsfile *f, struct cyclic_work *w, int *status);

/* Load one periodic spectrum from datafile */
int cyclic_load_ps(fitsfile *f, PS *d, int idx, int *status);

/* Init fft plans for datatype conversion */
int cyclic_init_ffts(struct cyclic_work *w);
void cyclic_free_ffts(struct cyclic_work *w);

/* Add polarizations in-place to get total intensity, 
 * allowing x/y gain factors to be applied */
int cyclic_pscrunch_ps(PS *d, float xgain, float ygain);

/* Sum over freq in periodic spectrum */
int cyclic_fscrunch_ps(struct profile_phase *out, PS *in);

/* Conversion routines */
void cyclic_ps2cs(PS *in, CS *out, const struct cyclic_work *w);
void cyclic_cs2cc(CS *in, CC *out, const struct cyclic_work *w);
void cyclic_cc2cs(CC *in, CS *out, const struct cyclic_work *w);
void filter_time2freq(struct filter_time *in, struct filter_freq *out,
        const struct cyclic_work *w);
void filter_freq2time(struct filter_freq *in, struct filter_time *out,
        const struct cyclic_work *w);
void profile_phase2harm(struct profile_phase *in, struct profile_harm *out,
        const struct cyclic_work *w);
void profile_harm2phase(struct profile_harm *in, struct profile_phase *out,
        const struct cyclic_work *w);

/* Make "shifted" cyclic spectrum, ie S(alpha, nu - alpha/2) in-place */
int cyclic_shift_cs(CS *d, int sign, const struct cyclic_work *w);

/* Make shifted filter copies */
int filter_shift(struct filter_freq *out, struct filter_time *in, 
        int nshift, double dfreq, const struct cyclic_work *w);

/* Normalize profile and filter */
int filter_profile_norm(struct filter_time *f, struct profile_harm *p, 
        int max_harm);

/* Mean square diff for convergence tests */
double profile_ms_difference(struct profile_harm *p1, struct profile_harm *p2,
        int max_harm);
double filter_ms_difference(struct filter_time *f1, struct filter_time *f2);
double cyclic_mse(CS *cs_shifted_pos,  CS *cs_shifted_neg,
        struct profile_harm *s, 
        struct filter_freq *h_shift_array_pos, 
        struct filter_freq *h_shift_array_neg, 
        int max_harm);

#endif
