/* solve_cyclic.c
 *
 * Determine best "snapshot" H(nu) and descattered profile
 * using the NLOpt optimization library.
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
#include "model_cyclic.h"

#define fits_error_check_fatal() do { \
    if (status) { \
        fits_report_error(stderr, status); \
        exit(1); \
    } \
} while (0)

void usage() {
    printf("solve_cyclic_nlopt\n");
}

/* Struct for passing the data to nlopt */
struct cyclic_data {
    struct cyclic_spectrum *cs; /* The actual data */
    struct profile_harm *s0;    /* The model profile */
    struct filter_time *ht;     /* The model filter func */
    struct cyclic_spectrum *model_cs; /* The resulting model CS */
    struct cyclic_work *w;      /* FFTW plans, etc */
};

/* Return the mean square different between current model params and data
 * using the functional form that nlopt wants.  Vector "x" contains the
 * parameter values (S0, H).
 */
double cyclic_ms_difference_nlopt(int n, double *x, 
        double *grad, void *_data) {

    /* Pointer to input data */
    struct cyclic_data *data = (struct cyclic_data *)_data;

    /* check dimensions */
    if (n != 2*(data->s0->nharm-1) + 2*(data->ht->nlag)) {
        fprintf(stderr, 
                "cyclic_ms_difference_nlopt: error, inconsistent sizes!\n");
        exit(1);
    }

    /* We dont' know how to compute gradients yet */
    if (grad != NULL) {
        fprintf(stderr, 
                "cyclic_ms_difference_nlopt: gradient not supported yet\n");
        exit(1);
    }

    /* Convert input "x" vector to structs */
    int i, j;
    data->s0->data[0] = 0.0;
    double *xtmp = x;
    for (i=1; i<data->s0->nharm; i++) { 
        data->s0->data[i] = xtmp[0] + I*xtmp[1];
        xtmp += 2;
    }
    for (i=0; i<data->ht->nlag; i++) { 
        data->ht->data[i] = xtmp[0] + I*xtmp[1]; 
        xtmp += 2;
    }

    /* Compute resulting model cyclic spectrum */
    int rv;
    rv = model_cyclic(data->model_cs, data->s0, data->ht, NULL, NULL, data->w);
    if (rv != 0) {
        fprintf(stderr, 
                "cyclic_ms_difference_nlopt: error in model_cyclic (%d)\n",
                rv);
        exit(1);
    }

    /* Return mean square diff between model and data */
    return cyclic_ms_diff(data->cs, data->model_cs);
}

/* Catch sigint */
int run=1;
void cc(int sig) { run=0; }

int main(int argc, char *argv[]) {

    int opt=0, verb=0;
    int max_harm = 64, max_lag=0;
    while ((opt=getopt(argc,argv,"hvH:L:"))!=-1) {
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
    struct cyclic_spectrum cs, model_cs;
    struct filter_time ht;
    struct filter_freq hf;
    struct profile_phase pp;
    struct profile_harm ph;

    raw.nphase = pp.nphase = w.nphase;
    raw.nchan = cs.nchan = hf.nchan = w.nchan;
    cs.nharm = ph.nharm =  w.nharm;
    ht.nlag = w.nlag;
    raw.npol = orig_npol;
    cs.npol = 1;

    model_cs.nchan = cs.nchan;
    model_cs.nharm = cs.nharm;
    model_cs.npol = cs.npol;

    cyclic_alloc_ps(&raw);
    cyclic_alloc_cs(&cs);
    cyclic_alloc_cs(&model_cs);
    filter_alloc_time(&ht);
    filter_alloc_freq(&hf);
    profile_alloc_phase(&pp);
    profile_alloc_harm(&ph);

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

    /* convert input data to cyclic spectrum */
    cyclic_ps2cs(&raw, &cs, &w);

    /* TODO output initial profile? */

    /* Fill in data struct for nlopt */
    struct cyclic_data cdata;
    cdata.cs = &cs;
    cdata.s0 = &ph;
    cdata.ht = &ht;
    cdata.model_cs = &model_cs;


    /* All done :) */
    exit(0);
}
