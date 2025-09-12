#define SHMEM_LOG_NAME "NW_LOG"
#define SHMEM_LOG_MAX 2048
#define MAX_BOXES 2048
#define MAX_ACTUATORS 256
#define MAX_IMAGE_SIZE 4194304
struct shmem_log_entry {
  uint8_t frame_number;
  uint8_t total_frame_number;
  double time0;
  double time1;
  double time2;

  uint8_t im[MAX_IMAGE_SIZE];
  double centroid_x[MAX_BOXES];
  double centroid_y[MAX_BOXES];
  double mirrors[MAX_ACTUATORS];
};
