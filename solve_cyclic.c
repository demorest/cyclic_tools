/* solve_cyclic.c
 *
 * Determine best "snapshot" H(nu) and descattered profile.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <getopt.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <fitsio.h>

#include "cyclic_utils.h"

#define fits_error_check_fatal() do { \
    if (status) { \
        fits_report_error(stderr, status); \
        exit(1); \
    } \
} while (0)

void usage() {
    printf("solve_cyclic\n");
}

void write_profile(const char *fname, struct profile_phase *p) {
    FILE *f = fopen(fname, "a");
    int i;
    for (i=0; i<p->nphase; i++) {
        fprintf(f,"%.7e %.7e\n", (double)i/(double)p->nphase, p->data[i]);
    }
    fprintf(f,"\n\n");
    fclose(f);
}

void write_fprofile(const char *fname, struct profile_harm *p) {
    FILE *f = fopen(fname, "a");
    int i;
    for (i=0; i<p->nharm; i++) {
        fprintf(f,"%d %.7e %.7e\n", i, creal(p->data[i]), cimag(p->data[i]));
    }
    fprintf(f,"\n\n");
    fclose(f);
}

void write_filter(const char *fname, struct filter_time *h) {
    FILE *f = fopen(fname, "a");
    int i;
    for (i=0; i<h->nlag; i++) {
        fprintf(f,"%d %.7e %.7e\n", i, creal(h->data[i]), cimag(h->data[i]));
    }
    fprintf(f,"\n\n");
    fclose(f);
}

void write_filter_freq(const char *fname, struct filter_freq *h) {
    FILE *f = fopen(fname, "a");
    int i;
    for (i=0; i<h->nchan; i++) {
        fprintf(f,"%d %.7e %.7e\n", i, creal(h->data[i]), cimag(h->data[i]));
    }
    fprintf(f,"\n\n");
    fclose(f);
}




/* Catch sigint */
int run=1;
void cc(int sig) { run=0; }

int main(int argc, char *argv[]) {

    int opt=0, verb=0;
    int max_harm = 64, max_lag=0;
    int causal_filter = 0;
    while ((opt=getopt(argc,argv,"hvH:L:C"))!=-1) {
        switch (opt) {
            case 'v':
                verb++;
                break;
            case 'H':
                max_harm = atoi(optarg);
                break;
            case 'L':
                max_lag = atoi(optarg);
                break;
            case 'C':
                causal_filter = 1;
                break;
            case 'h':
                usage();
                exit(0);
                break;
        }
    }

    if (optind==argc) {
        usage();
        exit(1);
    }

    int i, rv;

    /* Open file */
    fitsfile *f;
    int status;
    fits_open_file(&f, argv[optind], READONLY, &status);
    fits_error_check_fatal();

    /* Get basic dims */
    struct cyclic_work w;
    cyclic_load_params(f, &w, &status);
    fits_error_check_fatal();
    if (verb) { 
        printf("Read nphase=%d npol=%d nchan=%d\n", 
                w.nphase, w.npol, w.nchan);
        fflush(stdout);
    }
    int orig_npol = w.npol;
    w.npol = 1;

    /* Init FFTs */
    fftwf_init_threads();
    fftwf_plan_with_nthreads(4);
    if (verb) { printf("Planning FFTs\n"); fflush(stdout); }
#define WF "/home/pdemores/share/cyclic_wisdom.dat"
    FILE *wf = fopen(WF,"r");
    if (wf!=NULL) { fftwf_import_wisdom_from_file(wf); fclose(wf); }
    rv = cyclic_init_ffts(&w);
    if (rv) {
        fprintf(stderr, "Error planning ffts (rv=%d)\n", rv);
        exit(1);
    }
    wf = fopen(WF,"w");
    if (wf!=NULL) { fftwf_export_wisdom_to_file(wf); fclose(wf); }

    /* Alloc some stuff */
    struct periodic_spectrum raw;
    struct cyclic_spectrum cs, cs_neg;
    struct filter_time ht, ht_new;
    struct filter_freq hf, hf_new;
    struct filter_freq *hf_shift_pos, *hf_shift_neg;
    hf_shift_pos = (struct filter_freq *)malloc(
            sizeof(struct filter_freq)*w.nharm);
    hf_shift_neg = (struct filter_freq *)malloc(
            sizeof(struct filter_freq)*w.nharm);
    struct profile_phase pp, pp_new;
    struct profile_harm ph, ph_new;

    raw.nphase = pp.nphase = pp_new.nphase = w.nphase;
    raw.nchan = cs.nchan = hf.nchan = hf_new.nchan = w.nchan;
    cs.nharm = ph.nharm = ph_new.nharm = w.nharm;
    ht.nlag = ht_new.nlag = w.nlag;
    for (i=0; i<w.nharm; i++) { hf_shift_pos[i].nchan = w.nchan; }
    for (i=0; i<w.nharm; i++) { hf_shift_neg[i].nchan = w.nchan; }
    raw.npol = orig_npol;
    cs.npol = 1;

    cs_neg.nchan = cs.nchan;
    cs_neg.nharm = cs.nharm;
    cs_neg.npol = cs.npol;

    cyclic_alloc_ps(&raw);
    cyclic_alloc_cs(&cs);
    cyclic_alloc_cs(&cs_neg);
    ht.data = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*ht.nlag);
    ht_new.data = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*ht.nlag);
    hf.data = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*hf.nchan);
    hf_new.data = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*hf.nchan);
    for (i=0; i<w.nharm; i++) {
        hf_shift_pos[i].data = (fftwf_complex *)fftwf_malloc(
                sizeof(fftwf_complex)*hf.nchan);
        hf_shift_neg[i].data = (fftwf_complex *)fftwf_malloc(
                sizeof(fftwf_complex)*hf.nchan);
    }
    pp.data = (float *)fftwf_malloc(sizeof(float)*pp.nphase);
    pp_new.data = (float *)fftwf_malloc(sizeof(float)*pp.nphase);
    ph.data = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*ph.nharm);
    ph_new.data = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*ph.nharm);

    /* Check bounds */
    if (max_harm > w.nharm) { max_harm = w.nharm; }
    if (max_lag > w.nlag/2) { max_lag = w.nlag/2; }
    if (verb) {
        printf("Using max of %d harmonics and %d lags\n", max_harm, max_lag);
    }

    /* Run procedure on subint 0 */
    int isub = 1;

    /* Load data */
    cyclic_load_ps(f, &raw, isub, &status);
    fits_error_check_fatal();

    /* Add polns w/o calibration */
    cyclic_pscrunch_ps(&raw, 1.0, 1.0);

    /* Initialize H, profile guesses */
    cyclic_fscrunch_ps(&pp, &raw);
    profile_phase2harm(&pp, &ph, &w);
    ht.data[0] = 1.0;
    for (i=1; i<ht.nlag; i++) { ht.data[i] = 0.0; }
    filter_profile_norm(&ht, &ph, max_harm);
    profile_harm2phase(&ph, &pp, &w);

    /* convert to CS, produce shifted version */
    cyclic_ps2cs(&raw, &cs, &w);
    cyclic_ps2cs(&raw, &cs_neg, &w);
    cyclic_shift_cs(&cs, +1, &w);
    cyclic_shift_cs(&cs_neg, -1, &w);

    /* TODO output initial profile */

    /* Remove old files */
#define FILT "filters.dat"
#define TFILT "tfilters.dat"
#define PROF "profs.dat"
#define FPROF "fprofs.dat"
    unlink(FILT);
    unlink(TFILT);
    unlink(PROF);
    unlink(FPROF);

    FILE *it = fopen("iter.dat", "w");

    /* iterate */
    int nit=0;
    double mse=0.0, last_mse=0.0;
    signal(SIGINT, cc);
    do { 

        if (verb) {
            printf("iter %d\n", nit); 
            fflush(stdout);
        }

        /* Make freq domain filter */
        filter_time2freq(&ht, &hf, &w);
        write_filter(TFILT, &ht);
        write_filter_freq(FILT, &hf);

        /* Make shifted filter array */
        filter_shift(hf_shift_pos, &ht, w.nharm, 
                raw.ref_freq/(raw.bw*1e6), &w);
        filter_shift(hf_shift_neg, &ht, w.nharm, 
                -1.0*raw.ref_freq/(raw.bw*1e6), &w);

        mse = cyclic_mse(&cs, &cs_neg, &ph, hf_shift_pos, hf_shift_neg, 
                max_harm);

        /* Update filter, prof */
        cyclic_update_filter(&hf_new, &cs, &cs_neg, &ph, 
                hf_shift_pos, hf_shift_neg, max_harm);
        cyclic_update_profile(&ph_new, &cs, &cs_neg, 
                hf_shift_pos, hf_shift_neg);


        /* Back to time domain filter */
        filter_freq2time(&hf_new, &ht_new, &w);

        /* Fix filter normalization */
        for (i=0; i<ht_new.nlag; i++) 
            ht_new.data[i] /= (float)ht_new.nlag;

        /* Zero out negative lags */
        if (causal_filter) {
            for (i=ht_new.nlag/2; i<ht_new.nlag; i++) 
                ht_new.data[i] = 0.0;
        }
        
        /* Zero out large lags */
        if (max_lag>0) { 
            for (i=max_lag; i<ht_new.nlag-max_lag; i++) 
                ht_new.data[i] = 0.0;
        }

        /* Kill nyquist point?? */
        ht_new.data[ht_new.nlag/2] = 0.0;

        /* Normalize prof and filter */
        filter_profile_norm(&ht_new, &ph_new, max_harm);

        /* TODO some kind of convergence test */
        double prof_diff = profile_ms_difference(&ph, &ph_new, max_harm);
        double filt_diff = filter_ms_difference(&ht, &ht_new);

        /* TODO zero out high harmonics ?? */

        /* Step halfway to new versions, except first time */
        if (nit==0) {
            for (i=0; i<w.nharm; i++) 
                ph.data[i] = ph_new.data[i];
            for (i=0; i<w.nlag; i++) 
                ht.data[i] = ht_new.data[i]; 
        } else {
            //double fac = (mse<last_mse) ? 1.0 : 0.5*sqrt(mse/last_mse);
            double fac=0.25;
            for (i=0; i<w.nharm; i++) 
                ph.data[i] = (1.0-fac)*ph.data[i] + fac*ph_new.data[i];
            for (i=0; i<w.nlag; i++) 
                ht.data[i] = (1.0-fac)*ht.data[i] + fac*ht_new.data[i]; 
        }

        /* Back to phase domain profile */
        ph.data[0] = 0.0;
        profile_harm2phase(&ph, &pp_new, &w);

        /* Write out current profiles */
        write_profile(PROF, &pp_new);
        write_fprofile(FPROF, &ph);

        /* Print convergence params */
        if (verb) {
            fprintf(it,"%.3e %.3e %.8e %.8e\n", prof_diff, filt_diff, mse,
                    mse - last_mse);
        }
        last_mse = mse;

        /* Update iter count */
        nit++;

    } while (run);

    fclose(it);

    exit(0);

}
