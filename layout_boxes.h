#define MAX_BOXES 1024
#define MAX_TERMS 67108864
#define MAX_MIRROR_VOLTAGES 256

struct shmem_boxes_header
{
	uint8_t header_version=NW_HEADER_VERSION;
	uint8_t lock=0;
	uint16_t num_boxes;

	double pixel_um;
	double box_um;
	double pupil_radius_um;

	uint16_t nTerms; // Influence Fn
	uint16_t nActuators;

	float reference_x[MAX_BOXES];
	float reference_y[MAX_BOXES];
	float box_x[MAX_BOXES];
	float box_y[MAX_BOXES];
	float centroid_x[MAX_BOXES];
	float centroid_y[MAX_BOXES];
	float box_x_normalized[MAX_BOXES];
	float box_y_normalized[MAX_BOXES];
  float delta_x[MAX_BOXES];
  float delta_y[MAX_BOXES];

  double influence[MAX_TERMS];
  double influence_inv[MAX_TERMS];

  double mirror_voltages[MAX_MIRROR_VOLTAGES];
};

#define SHMEM_BUFFER_SIZE_BOXES 1+1+2+sizeof(double)*3+MAX_BOXES*sizeof(float)*10+sizeof(double)*MAX_TERMS*2
#define SHMEM_BUFFER_NAME_BOXES "NW_BUFFER_BOXES"
