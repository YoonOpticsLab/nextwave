#define NW_HEADER_VERSION 1

#define NW_STATUS_READ 1

#define NW_UINT8 1

#define MAX_IMAGE_SIZE 2048*2048
#define NW_MAX_FRAMES 4

#define SHMEM_HEADER_SIZE 512
#define SHMEM_HEADER_NAME "NW_SRC0_HDR"

#define SHMEM_BUFFER_SIZE MAX_IMAGE_SIZE*NW_MAX_FRAMES
#define SHMEM_BUFFER_NAME "NW_SRC0_BUFFER"

#define MODE_OFF 0
#define MODE_READY 1
#define MODE_SNAP 2
#define MODE_RUN  4
#define MODE_AO   8
#define MODE_FORCE_AO_START 0x20
#define MODE_FORCE_AO_DURING 0x40
#define MODE_CALIBRATING 0x80
#define MODE_QUIT 0xFF

struct shmem_header
{
	uint8_t lock=0;
	uint8_t header_version=NW_HEADER_VERSION;
	uint8_t mode=MODE_OFF;
	uint16_t manager_port_num;
	uint16_t dimensions[4];
	uint16_t datatype_code;
	uint16_t fps[3];
	uint8_t num_ring_frames;
	uint8_t current_frame;
	uint64_t total_frames;
	uint64_t frames_left;
	uint64_t log_index;	
	uint8_t max_frames=NW_MAX_FRAMES;
	uint64_t timestamps[NW_MAX_FRAMES];
	uint64_t statuses[NW_MAX_FRAMES];
};
