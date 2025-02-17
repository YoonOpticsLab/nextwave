#define SHMEM_LOG_NAME "NW_LOG"
#define SHMEM_LOG_MAX 4096
#define MAX_BOXES 2048
#define MAX_ACTUATORS 1024
struct shmem_log_entry {
  uint8_t frame_number;
  uint8_t total_frame_number;
  double time0;
  double time1;
  double time2;

  double centroid_x[MAX_BOXES];
  double centroid_y[MAX_BOXES];
  double mirrors[MAX_ACTUATORS];
};
