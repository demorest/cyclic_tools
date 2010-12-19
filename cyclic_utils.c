/* cyclic_utils.c 
 *
 * P. Demorest, 2010
 *
 */

#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <fitsio.h>

#include "cyclic_utils.h"

/* Allocs / frees */
void cyclic_alloc_ps(PS *d) {
    d->data = (float *)fftwf_malloc(sizeof(float) * 
            d->nphase * d->nchan * d->npol);
}
void cyclic_alloc_cs(CS *d) {
    d->data = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * 
            d->nharm * d->nchan * d->npol);
}
void cyclic_alloc_cc(CC *d) {
    d->data = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * 
            d->nharm * d->nlag * d->npol);
}
void cyclic_alloc_pc(PC *d) {
    d->data = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * 
            d->nphase * d->nlag * d->npol);
}
void cyclic_free_ps(PS *d) { fftwf_free(d->data); }
void cyclic_free_cs(CS *d) { fftwf_free(d->data); }
void cyclic_free_cc(CC *d) { fftwf_free(d->data); }
void cyclic_free_pc(PC *d) { fftwf_free(d->data); }

void filter_alloc_time(struct filter_time *f) {
    f->data = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) *
            f->nlag);
}
void filter_alloc_freq(struct filter_freq *f) {
    f->data = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) *
            f->nchan);
}
void filter_free_time(struct filter_time *f) { fftwf_free(f->data); }
void filter_free_freq(struct filter_freq *f) { fftwf_free(f->data); }

void profile_alloc_phase(struct profile_phase *f) { 
    f->data = (float *)fftwf_malloc(sizeof(float) * f->nphase); 
}
void profile_alloc_harm(struct profile_harm *f) { 
    f->data = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * f->nharm); 
}
void profile_free_phase(struct profile_phase *f) { fftwf_free(f->data); }
void profile_free_harm(struct profile_harm *f) { fftwf_free(f->data); }

/* Load dimension params from fits file */
int cyclic_load_params(fitsfile *f, struct cyclic_work *w, int *status) {

    int bitpix, naxis; 
    long naxes[4];

    fits_get_img_param(f, 4, &bitpix, &naxis, naxes, status);
    if (naxis!=4) { return(-1); }

    w->nphase = naxes[0];
    w->npol = naxes[1];
    w->nchan = naxes[2];

    w->nlag = 0;
    w->nharm = 0;

    return(*status);
}

/* Load one periodic spectrum from datafile 
 * Space should already be allocated.
 * idx is 1-offset following cfitsio convention.
 */
int cyclic_load_ps(fitsfile *f, PS *d, int idx, int *status) {

    /* Load data */
    long fpixel[4];
    long long nelem;
    fpixel[0] = fpixel[1] = fpixel[2] = 1;
    fpixel[3] = idx;
    nelem = d->nphase * d->npol * d->nchan;
    fits_read_pix(f, TFLOAT, fpixel, nelem, NULL, d->data, NULL, status);

    /* Load header params */
    char key[9];
    sprintf(key, "IMJD%04d", idx);
    fits_read_key(f, TINT, key, &d->imjd, NULL, status);
    sprintf(key, "FMJD%04d", idx);
    fits_read_key(f, TDOUBLE, key, &d->fmjd, NULL, status);
    sprintf(key, "PHAS%04d", idx);
    fits_read_key(f, TDOUBLE, key, &d->ref_phase, NULL, status);
    sprintf(key, "FREQ%04d", idx);
    fits_read_key(f, TDOUBLE, key, &d->ref_freq, NULL, status);
    // TODO get these in the file
    d->rf = 428.0;
    d->bw = 4.0;

    return(*status);
}

/* Set up fft plans.  Need to have npol, nphase, nchan 
 * already filled in struct */
int cyclic_init_ffts(struct cyclic_work *w) {

    /* Infer lag, harmonic sizes from chan/phase */
    w->nlag = w->nchan; // Total number of lags including + and -
    w->nharm = w->nphase/2 + 1; // Only DC and positive harmonics

    /* Alloc temp arrays for planning */
    PS ps;
    CS cs;
    CC cc;
    PC pc;
    struct filter_time ft;
    struct filter_freq ff;
    struct profile_phase pp;
    struct profile_harm ph;

    ps.npol = cs.npol = cc.npol = pc.npol = w->npol;
    ps.nphase = pc.nphase = w->nphase;
    ps.nchan = cs.nchan = w->nchan;
    cs.nharm = cc.nharm = w->nharm;
    cc.nlag = pc.nlag = w->nlag;

    cyclic_alloc_ps(&ps);
    cyclic_alloc_cs(&cs);
    cyclic_alloc_cc(&cc);
    cyclic_alloc_pc(&pc);

    ft.nlag = w->nlag;
    ff.nchan = w->nchan;
    pp.nphase = w->nphase;
    ph.nharm = w->nharm;

    filter_alloc_time(&ft);
    filter_alloc_freq(&ff);
    profile_alloc_phase(&pp);
    profile_alloc_harm(&ph);

    /* FFT plans */
    int rv=0;

    /* ps2cs - r2c fft along phase (fastest) axis */
    w->ps2cs = fftwf_plan_many_dft_r2c(1, &w->nphase, w->npol*w->nchan,
            ps.data, NULL, 1, w->nphase,
            cs.data, NULL, 1, w->nharm,
            FFTW_MEASURE | FFTW_PRESERVE_INPUT);
    if (w->ps2cs == NULL) rv++; 

    /* cs2cc - c2c ifft along channel axis */
    w->cs2cc = fftwf_plan_many_dft(1, &w->nchan, w->npol*w->nharm,
            cs.data, NULL, w->nharm*w->npol, 1,
            cc.data, NULL, w->nharm*w->npol, 1,
            FFTW_BACKWARD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);
    if (w->cs2cc == NULL) rv++; 
    
    /* cc2cs - c2c fft along lag axis */
    w->cc2cs = fftwf_plan_many_dft(1, &w->nlag, w->npol*w->nharm,
            cc.data, NULL, w->nharm*w->npol, 1,
            cs.data, NULL, w->nharm*w->npol, 1,
            FFTW_FORWARD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);
    if (w->cc2cs == NULL) rv++; 

    /* time2freq, freq2time for filters */
    w->time2freq = fftwf_plan_dft_1d(w->nlag, ft.data, ff.data,
            FFTW_FORWARD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);
    if (w->time2freq == NULL) rv++;
    w->freq2time = fftwf_plan_dft_1d(w->nchan, ff.data, ft.data,
            FFTW_BACKWARD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);
    if (w->freq2time == NULL) rv++;

    /* phase2harm, harm2phase for profiles */
    w->phase2harm = fftwf_plan_dft_r2c_1d(w->nphase, pp.data, ph.data, 
            FFTW_MEASURE | FFTW_PRESERVE_INPUT);
    if (w->phase2harm == NULL) rv++;
    w->harm2phase = fftwf_plan_dft_c2r_1d(w->nphase, ph.data, pp.data, 
            FFTW_MEASURE | FFTW_PRESERVE_INPUT);
    if (w->harm2phase == NULL) rv++;

    cyclic_free_ps(&ps);
    cyclic_free_cs(&cs);
    cyclic_free_cc(&cc);
    cyclic_free_pc(&pc);

    filter_free_time(&ft);
    filter_free_freq(&ff);
    profile_free_phase(&pp);
    profile_free_harm(&ph);

    return(rv);
}

void cyclic_free_ffts(struct cyclic_work *w) {
    if (w->ps2cs!=NULL) fftwf_destroy_plan(w->ps2cs);
    if (w->cs2cc!=NULL) fftwf_destroy_plan(w->cs2cc);
    if (w->cc2cs!=NULL) fftwf_destroy_plan(w->cc2cs);
}

int cyclic_pscrunch_ps(PS *d, float xgain, float ygain) {

    if (d->npol<2) { return(-1); }

    int ichan, iphase;
    float *xx, *yy;
    for (ichan=0; ichan<d->nchan; ichan++) {
        for (iphase=0; iphase<d->nphase; iphase++) {
            xx = get_ps(d, iphase, 0, ichan);
            yy = get_ps(d, iphase, 1, ichan);
            *xx = xgain * (*xx) + ygain * (*yy);
        }
    }

    d->npol = 1;
    return(0);
}

int cyclic_fscrunch_ps(struct profile_phase *out, PS *in) {

    /* Only 1 pol for now */
    if (in->npol>1) return(-1); 

    int iphase, ichan;
    for (iphase=0; iphase<in->nphase; iphase++) {
        out->data[iphase] = 0.0;
        for (ichan=0; ichan<in->nchan; ichan++) {
            const float *tmp = get_ps(in,iphase,0,ichan);
            out->data[iphase] += *tmp;
        }
        out->data[iphase] /= (float)in->nchan;
    }

    return(0);
}

void cyclic_ps2cs(PS *in, CS *out, const struct cyclic_work *w) {
    fftwf_execute_dft_r2c(w->ps2cs, in->data, out->data);
    out->imjd = in->imjd;
    out->fmjd = in->fmjd;
    out->ref_phase = in->ref_phase;
    out->ref_freq = in->ref_freq;
    out->rf = in->rf;
    out->bw = in->bw;
}
void cyclic_cs2cc(CS *in, CC *out, const struct cyclic_work *w) {
    fftwf_execute_dft(w->cs2cc, in->data, out->data);
    out->imjd = in->imjd;
    out->fmjd = in->fmjd;
    out->ref_phase = in->ref_phase;
    out->ref_freq = in->ref_freq;
    out->rf = in->rf;
    out->bw = in->bw;
}
void cyclic_cc2cs(CC *in, CS *out, const struct cyclic_work *w) {
    fftwf_execute_dft(w->cc2cs, in->data, out->data);
    out->imjd = in->imjd;
    out->fmjd = in->fmjd;
    out->ref_phase = in->ref_phase;
    out->ref_freq = in->ref_freq;
    out->rf = in->rf;
    out->bw = in->bw;
}

void filter_time2freq(struct filter_time *in, struct filter_freq *out, 
        const struct cyclic_work *w) {
    fftwf_execute_dft(w->time2freq, in->data, out->data);
}
void filter_freq2time(struct filter_freq *in, struct filter_time *out, 
        const struct cyclic_work *w) {
    fftwf_execute_dft(w->freq2time, in->data, out->data);
}
void profile_phase2harm(struct profile_phase *in, struct profile_harm *out, 
        const struct cyclic_work *w) {
    fftwf_execute_dft_r2c(w->phase2harm, in->data, out->data);
}
void profile_harm2phase(struct profile_harm *in, struct profile_phase *out, 
        const struct cyclic_work *w) {
    fftwf_execute_dft_c2r(w->harm2phase, in->data, out->data);
}

int cyclic_shift_cs(CS *d, int sign, const struct cyclic_work *w) {

    CC tmp_cc;
    tmp_cc.nharm = d->nharm;
    tmp_cc.npol = d->npol;
    tmp_cc.nlag = d->nchan;
    cyclic_alloc_cc(&tmp_cc);

    const double dtau = 1.0 / (d->bw*1e6);  // lag step, in seconds
    const double dalpha = d->ref_freq;  // harmonic step in Hz

    /* Move to cc domain */
    cyclic_cs2cc(d, &tmp_cc, w);

    /* Multiply by shift function */
    int iharm, ilag, ipol;
    for (ilag=0; ilag<tmp_cc.nlag; ilag++) {
        for (iharm=0; iharm<tmp_cc.nharm; iharm++) {
            // TODO check sign
            int lag = (ilag<=tmp_cc.nlag/2) ? ilag : ilag-tmp_cc.nlag;
            double phs = 2.0*M_PI*(dalpha*(double)iharm/2.0) * 
                (dtau*(double)lag);
            fftwf_complex fac = (cos(phs)+I*sin(phs))/(float)w->nchan;
            if (sign<0) fac = conj(fac);
            for (ipol=0; ipol<tmp_cc.npol; ipol++) {
                fftwf_complex *dat = get_cc(&tmp_cc,iharm,ipol,ilag);
                *dat *= fac;
            }
        } 
    }

    /* Back to orig domain */
    cyclic_cc2cs(&tmp_cc, d, w);

    cyclic_free_cc(&tmp_cc);
    return(0);
}

int filter_shift(struct filter_freq *out, struct filter_time *in,
        int nshift, double dfreq, 
        const struct cyclic_work *w) {
        

    struct filter_freq *cur;
    struct filter_time tmp;
    tmp.nlag = in->nlag;
    filter_alloc_time(&tmp);

    int ishift, ilag;
    for (ishift=0; ishift<nshift; ishift++) {
        cur = &out[ishift];
        for (ilag=0; ilag<in->nlag; ilag++) {
            // TODO check sign, normalization
            int lag = (ilag<=in->nlag/2) ? ilag : ilag - in->nlag;
            double phs = 2.0*M_PI*(double)ishift*(double)lag*dfreq;
            fftwf_complex fac = (cos(phs)+I*sin(phs));
            tmp.data[ilag] = in->data[ilag] * fac;
        }
        filter_time2freq(&tmp, cur, w);

#if 0 
        /* Zero out wraparound points */
        for (ichan=0; ichan<cur->nchan; ichan++) 
            if ((double)ichan/(double)cur->nchan < (double)ishift*dfreq) 
                cur->data[ichan] = 0.0;
#endif

    }

    filter_free_time(&tmp);
    return(0);
}

int filter_profile_norm(struct filter_time *f, struct profile_harm *p,
        int max_harm) {

    double psum = 0.0;
    int i;
    for (i=1; i<max_harm; i++) 
        psum += creal(p->data[i]*conj(p->data[i]));
    psum = sqrt(psum);

    for (i=0; i<p->nharm; i++) 
        p->data[i] /= psum;

    for (i=0; i<f->nlag; i++) 
        f->data[i] *= sqrt(psum);

    return(0);

}

double cyclic_mse(CS *cs_shifted_pos,  CS *cs_shifted_neg,
        struct profile_harm *s, 
        struct filter_freq *h_shift_array_pos, 
        struct filter_freq *h_shift_array_neg, 
        int max_harm) {

    /* Only valid for 1-pol data now */
    if (cs_shifted_pos->npol!=1) { return(-1); }

    if (max_harm<=0) max_harm = cs_shifted_pos->nharm;

    /* Loop sums over both nu and alpha */
    int iharm, ichan;
    double sum = 0.0;
    for (iharm=1; iharm<max_harm; iharm++) {

        fftwf_complex sa = s->data[iharm];

        for (ichan=0; ichan<cs_shifted_pos->nchan; ichan++) {

            /* Positive alpha */
            fftwf_complex *cs = get_cs(cs_shifted_pos,iharm,0,ichan);
            fftwf_complex h  = h_shift_array_pos[0].data[ichan];
            fftwf_complex ha = h_shift_array_pos[iharm].data[ichan];

            fftwf_complex diff = (*cs) - h*conj(ha)*sa;
            sum += creal(diff * conj(diff));

            /* negative alpha */
            cs = get_cs(cs_shifted_neg,iharm,0,ichan);
            h  = h_shift_array_neg[0].data[ichan];
            ha = h_shift_array_neg[iharm].data[ichan];

            diff = conj(*cs) - h*conj(ha)*conj(sa);
            sum += creal(diff * conj(diff));
        }
    }

    return(sum);

}

double cyclic_ms_difference (CS *cs1, CS *cs2) {
    double sum = 0.0;
    int ih, ic, ip;
    for (ic=0; ic<cs1->nchan; ic++) {
        for (ip=0; ip<cs1->npol; ip++) {
            for (ih=0; ih<cs1->nharm; ih++) {
                fftwf_complex *d1 = get_cs(cs1, ih, ip, ic);
                fftwf_complex *d2 = get_cs(cs2, ih, ip, ic);
                fftwf_complex diff = (*d1) - (*d2);
                sum += creal(diff*conj(diff));
            }
        }
    }
    return(sum);
}

double profile_ms_difference(struct profile_harm *p1, struct profile_harm *p2,
        int max_harm) {
    double sum = 0.0;
    int i;
    for (i=1; i<max_harm; i++) {
        fftwf_complex diff = p1->data[i] - p2->data[i];
        sum += creal(diff*conj(diff));
    }
    return(sum);
}

double filter_ms_difference(struct filter_time *f1, struct filter_time *f2) {
    double sum = 0.0;
    int i;
    for (i=1; i<f1->nlag; i++) {
        fftwf_complex diff = f1->data[i] - f2->data[i];
        sum += creal(diff*conj(diff));
    }
    return(sum);
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

