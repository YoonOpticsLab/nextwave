#define MAX_BOXES 1024
#define MAX_TERMS 67108864


struct shmem_boxes_header
{
	uint8_t header_version=NW_HEADER_VERSION;
	uint8_t lock=0;
	uint16_t num_boxes;

	double pixel_um;
	double box_um;
	double pupil_radius_um;

	float reference_x[MAX_BOXES];
	float reference_y[MAX_BOXES];
	float box_x[MAX_BOXES];
	float box_y[MAX_BOXES];
	float centroid_x[MAX_BOXES];
	float centroid_y[MAX_BOXES];
	float box_x_normalized[MAX_BOXES];
	float box_y_normalized[MAX_BOXES];

  double zterms[MAX_TERMS];
  double zterms_inv[MAX_TERMS];
};

#define SHMEM_BUFFER_SIZE_BOXES 1+1+2+sizeof(double)*3+MAX_BOXES*sizeof(float)+sizeof(double)*MAX_TERMS
#define SHMEM_BUFFER_NAME_BOXES "NW_BUFFER_BOXES"
