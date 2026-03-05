#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <assert.h>

// libraries to help w parallelization
#include <sched.h> // for processor affinity
#include <unistd.h>   // unix standard apis
#include <pthread.h>

#include "filter.h"
#include "signal.h"
#include "timing.h"

#define MAXWIDTH 40
#define THRESHOLD 2.0
#define ALIENS_LOW  50000.0
#define ALIENS_HIGH 150000.0


long num_processors; // number of processors to use
pthread_t* tid;  // array of thread ids
long num_threads; // number of threads
int num_bands; // number of bands


void usage() {
  printf("usage: band_scan text|bin|mmap signal_file Fs filter_order num_bands num_threads num_processors\n");
}

// struct to hold arguments for threads
struct thread_args {
  long id; // id of thread
  double bw; //bandwidth
  int filter_o; // filter order
  double* filter_c; // filter coeffs
  double* bp; // band power
  signal* sg; // sig
};

// function run by each thread
void* worker(void* arg) {
  struct thread_args* args = (struct thread_args*) arg;
  long myid = (long)args->id;
  double bandw = (double)args->bw;
  int f_o = (int)args->filter_o;
  double* f_c = (double*)args->filter_c;
  double* bandp = (double*)args->bp;
  signal* sgnl = (signal*)args->sg;
  // need to calculate blocksize using number of bands and number of threads
  int blocksize = num_bands / num_threads;

  // put ourselves on the desired processor
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(myid % num_processors, &set);
  if (sched_setaffinity(0, sizeof(set), &set) < 0) { // do it
    perror("Can't setaffinity"); // hopefully doesn't fail
    exit(-1);
  }


  // which band to start on based on id
  int mystart = myid * blocksize;
  int myend = 0;
  if (myid == (num_threads - 1)) { // last thread
    myend = num_bands;
  }
  else {
    myend = (myid + 1) * blocksize;
  }

  for (int band = mystart; band < myend; band++) { // parallelize this
    // Make the filter
    generate_band_pass(sgnl->Fs,
                       band * bandw + 0.0001, // keep within limits
                       (band + 1) * bandw - 0.0001,
                       f_o,
                       f_c);
    hamming_window(f_o,f_c);

    // Convolve
    convolve_and_compute_power(sgnl->num_samples,
                               sgnl->data,
                               f_o,
                               f_c,
                               &(bandp[band]));

  }

  pthread_exit(NULL);           // finish - no return value

}

double avg_power(double* data, int num) {

  double ss = 0;
  for (int i = 0; i < num; i++) {
    ss += data[i] * data[i];
  }

  return ss / num;
}

double max_of(double* data, int num) {

  double m = data[0];
  for (int i = 1; i < num; i++) {
    if (data[i] > m) {
      m = data[i];
    }
  }
  return m;
}

double avg_of(double* data, int num) {

  double s = 0;
  for (int i = 0; i < num; i++) {
    s += data[i];
  }
  return s / num;
}

void remove_dc(double* data, int num) {

  double dc = avg_of(data,num);

  printf("Removing DC component of %lf\n",dc);

  for (int i = 0; i < num; i++) {
    data[i] -= dc;
  }
}


int analyze_signal(signal* sig, int filter_order, int num_bands, double* lb, double* ub, long num_th, long num_p, pthread_t* tids) {

  double Fc        = (sig->Fs) / 2;
  double bandwidth = Fc / num_bands;

  remove_dc(sig->data,sig->num_samples);

  double signal_power = avg_power(sig->data,sig->num_samples);

  printf("signal average power:     %lf\n", signal_power);

  resources rstart;
  get_resources(&rstart,THIS_PROCESS);
  double start = get_seconds();
  unsigned long long tstart = get_cycle_count();

  double filter_coeffs[filter_order + 1];
  double band_power[num_bands];

  // create struct for thread args
  struct thread_args* args = (struct thread_args*) malloc(sizeof(struct thread_args)); 
  args->bw = bandwidth;
  args->bp = band_power;
  args->sg = sig;
  args->filter_o = filter_order;
  args->filter_c = filter_coeffs;


  for (long i = 0; i < num_th; i++) {
    args->id = (long)i;
    int returncode = pthread_create(&(tids[i]),
                                    NULL,
                                    worker,
                                    (void*)args
                                    );
    if (returncode != 0) {
      perror("Failed to start thread");
      exit(-1);
    }
  }

  // now we will join all the threads
  for (int i = 0; i < num_th; i++) {
    int returncode = pthread_join(tids[i], NULL);
    if (returncode != 0) {
      perror("join failed");
      exit(-1);
    }
  }

  // for (int band = 0; band < num_bands; band++) { // parallelize this
  //   // Make the filter
  //   generate_band_pass(sig->Fs,
  //                      band * bandwidth + 0.0001, // keep within limits
  //                      (band + 1) * bandwidth - 0.0001,
  //                      filter_order,
  //                      filter_coeffs);
  //   hamming_window(filter_order,filter_coeffs);

  //   // Convolve
  //   convolve_and_compute_power(sig->num_samples,
  //                              sig->data,
  //                              filter_order,
  //                              filter_coeffs,
  //                              &(band_power[band]));

  // }

  unsigned long long tend = get_cycle_count();
  double end = get_seconds();

  resources rend;
  get_resources(&rend,THIS_PROCESS);

  resources rdiff;
  get_resources_diff(&rstart, &rend, &rdiff);

  // Pretty print results
  double max_band_power = max_of(band_power,num_bands);
  double avg_band_power = avg_of(band_power,num_bands);
  int wow = 0;
  *lb = -1;
  *ub = -1;

  for (int band = 0; band < num_bands; band++) { 
    double band_low  = band * bandwidth + 0.0001;
    double band_high = (band + 1) * bandwidth - 0.0001;

    printf("%5d %20lf to %20lf Hz: %20lf ",
           band, band_low, band_high, band_power[band]);

    for (int i = 0; i < MAXWIDTH * (band_power[band] / max_band_power); i++) {
      printf("*");
    }

    if ((band_low >= ALIENS_LOW && band_low <= ALIENS_HIGH) ||
        (band_high >= ALIENS_LOW && band_high <= ALIENS_HIGH)) {

      // band of interest
      if (band_power[band] > THRESHOLD * avg_band_power) {
        printf("(WOW)");
        wow = 1;
        if (*lb < 0) {
          *lb = band * bandwidth + 0.0001;
        }
        *ub = (band + 1) * bandwidth - 0.0001;
      } else {
        printf("(meh)");
      }
    } else {
      printf("(meh)");
    }

    printf("\n");
  }

  printf("Resource usages:\n\
          User time        %lf seconds\n\
          System time      %lf seconds\n\
          Page faults      %ld\n\
          Page swaps       %ld\n\
          Blocks of I/O    %ld\n\
          Signals caught   %ld\n\
          Context switches %ld\n",
         rdiff.usertime,
         rdiff.systime,
         rdiff.pagefaults,
         rdiff.pageswaps,
         rdiff.ioblocks,
         rdiff.sigs,
         rdiff.contextswitches);

  printf("Analysis took %llu cycles (%lf seconds) by cycle count, timing overhead=%llu cycles\n"
         "Note that cycle count only makes sense if the thread stayed on one core\n",
         tend - tstart, cycles_to_seconds(tend - tstart), timing_overhead());
  printf("Analysis took %lf seconds by basic timing\n", end - start);

  return wow;
}

int main(int argc, char* argv[]) {

  if (argc != 8) { // need original 6 args plus 2 args for nums processors/threads
    usage();
    return -1;
  }

  char sig_type    = toupper(argv[1][0]);
  char* sig_file   = argv[2];
  double Fs        = atof(argv[3]);
  int filter_order = atoi(argv[4]);
  num_bands    = atoi(argv[5]);
  // new args for processors/threads
  num_threads = atoi(argv[6]); // number of threads
  printf("Num threads: %ld\n", num_threads);
  num_processors = atoi(argv[7]);  // number of processors to use

  tid = (pthread_t*)malloc(sizeof(pthread_t) * num_threads); // contains thread ids

  assert(Fs > 0.0);
  assert(filter_order > 0 && !(filter_order & 0x1));
  assert(num_bands > 0);

  printf("type:     %s\n\
          file:     %s\n\
          Fs:       %lf Hz\n\
          order:    %d\n\
          bands:    %d\n",
         sig_type == 'T' ? "Text" : (sig_type == 'B' ? "Binary" : (sig_type == 'M' ? "Mapped Binary" : "UNKNOWN TYPE")),
         sig_file,
         Fs,
         filter_order,
         num_bands);

  printf("Load or map file\n");

  signal* sig;
  switch (sig_type) {
    case 'T':
      sig = load_text_format_signal(sig_file);
      break;

    case 'B':
      sig = load_binary_format_signal(sig_file);
      break;

    case 'M':
      sig = map_binary_format_signal(sig_file);
      break;

    default:
      printf("Unknown signal type\n");
      return -1;
  }

  if (!sig) {
    printf("Unable to load or map file\n");
    return -1;
  }

  sig->Fs = Fs;

  double start = 0;
  double end   = 0;
  if (analyze_signal(sig, filter_order, num_bands, &start, &end, num_threads, num_processors, tid)) {
    printf("POSSIBLE ALIENS %lf-%lf HZ (CENTER %lf HZ)\n", start, end, (end + start) / 2.0);
  } else {
    printf("no aliens\n");
  }

  free_signal(sig);

  return 0;
}

