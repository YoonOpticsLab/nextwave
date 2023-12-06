#define MAX_BOXES 1024

struct shmem_boxes_header
{
	uint8_t header_version=NW_HEADER_VERSION;
	uint8_t lock=0;
	uint16_t num_boxes;
	double box_size;
	double pupil_radius_pixels;

	float reference_x[MAX_BOXES];
	float reference_y[MAX_BOXES];
	float box_x[MAX_BOXES];
	float box_y[MAX_BOXES];
	float centroid_x[MAX_BOXES];
	float centroid_y[MAX_BOXES];
};
