#define NW_HEADER_VERSION 1

#define NW_STATUS_READ 1

#define NW_UINT8 1

#define MAX_IMAGE_SIZE 2048*2048
#define NW_MAX_FRAMES 32

#define SHMEM_HEADER_SIZE 512
#define SHMEM_HEADER_NAME "NW_SRC0_HDR"

#define SHMEM_BUFFER_SIZE MAX_IMAGE_SIZE*NW_MAX_FRAMES
#define SHMEM_BUFFER_NAME "NW_SRC0_BUFFER"

#define SHMEM_BUFFER_SIZE2 65536
#define SHMEM_BUFFER_NAME2 "NW_BUFFER2"

struct shmem_header
{
	uint8_t lock=0;
	uint8_t header_version=NW_HEADER_VERSION;
	uint16_t manager_port_num;
	uint16_t dimensions[4];
	uint16_t datatype_code;
	uint8_t num_ring_frames;
	uint8_t current_frame;
	uint8_t max_frames=NW_MAX_FRAMES;
	uint64_t timestamps[NW_MAX_FRAMES];
	uint64_t statuses[NW_MAX_FRAMES];	
};
