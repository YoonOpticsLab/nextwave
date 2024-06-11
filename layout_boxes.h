#define MAX_BOXES 1024
#define MAX_MIRROR_VOLTAGES 256
#define MAX_TERMS 262144

#define CALC_TYPE double

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

	CALC_TYPE reference_x[MAX_BOXES];
	CALC_TYPE reference_y[MAX_BOXES];
	CALC_TYPE box_x[MAX_BOXES];
	CALC_TYPE box_y[MAX_BOXES];
	CALC_TYPE centroid_x[MAX_BOXES];
	CALC_TYPE centroid_y[MAX_BOXES];
	CALC_TYPE box_x_normalized[MAX_BOXES];
	CALC_TYPE box_y_normalized[MAX_BOXES];

  CALC_TYPE influence_inv[MAX_TERMS];
	
  CALC_TYPE delta_x[MAX_BOXES];
  CALC_TYPE delta_y[MAX_BOXES];

  CALC_TYPE mirror_voltages[MAX_MIRROR_VOLTAGES];
};

#define SHMEM_BUFFER_SIZE_BOXES 1+1+2+sizeof(CALC_TYPE)*3+4+MAX_BOXES*sizeof(CALC_TYPE)*10+sizeof(CALC_TYPE)*MAX_TERMS*2+MAX_MIRROR_VOLTAGES*sizeof(CALC_TYPE)
#define SHMEM_BUFFER_NAME_BOXES "NW_BUFFER_BOXES"
