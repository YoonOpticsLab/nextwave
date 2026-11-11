#define SOCKET_BUFFER_SIZE 1024 

//Example follows:

#include <string>
#include <cstring>
#include <boost/asio.hpp>
#include <boost/asio/read_until.hpp>
using boost::asio::ip::tcp;

class Connection {
  boost::asio::io_service ioService;
  std::shared_ptr<boost::asio::ip::tcp::socket> currentSocket;
  bool isConnected;
  char buffer[SOCKET_BUFFER_SIZE];
  char message[SOCKET_BUFFER_SIZE];
  std::string pending; // Bytes received but not yet returned as messages
  int port=0;

public:
  Connection(int port);
  ~Connection();

  int available() {return (int) currentSocket->available(); };
  char* read(int n_avail);
  char* next_message();
};

Connection::Connection(int port)
{
  this->port = port;
  currentSocket = std::make_shared<boost::asio::ip::tcp::socket>(ioService);
  tcp::acceptor a(ioService, boost::asio::ip::tcp::endpoint(tcp::v4(), port));
  a.accept( *currentSocket ); // Will block here and wait for client to connect
  isConnected = true;
}

Connection::~Connection()
{
  //currentSocket->disconnect();
}

char *Connection::read(int n_avail)
{
  boost::system::error_code error;
  if (n_avail > SOCKET_BUFFER_SIZE)
    n_avail = SOCKET_BUFFER_SIZE; // Rest stays in socket for next call
  size_t bytes_read = currentSocket->read_some(boost::asio::buffer(buffer, n_avail), error);
			   
  if (error == boost::asio::error::eof) {
    spdlog::error("SOCKET {} EOF", port);
  } else if (error) {
	spdlog::error("ERROR");
    throw boost::system::system_error(error); // Some other error
  } else {
//    spdlog::info("Message @{}: {} {}", port, bytes_read, buffer );
	pending.append(buffer, bytes_read);
  };

  return next_message();
}

// Messages are NUL-terminated, and several can arrive in one read (TCP coalesces
// quick sends, e.g. at startup). Return one complete message per call.
char *Connection::next_message()
{
  size_t end;
  while ((end = pending.find('\0')) != std::string::npos) {
    std::string msg = pending.substr(0, end);
    pending.erase(0, end+1);
    if (msg.empty())
      continue; // Skip empty messages (some senders add an extra NUL)
    strncpy(message, msg.c_str(), SOCKET_BUFFER_SIZE-1);
    message[SOCKET_BUFFER_SIZE-1] = 0;
    return message;
  }
  return NULL;
}

// GLOBAL (in context of includer)
Connection* theConnection=NULL;

// HELPER, uses globals
static char* socket_check(int port) {
if (theConnection==NULL) {
  //spdlog::info("Before New connection {}", port );
  // This blocks until receiver connects:
  theConnection = new Connection(port);
  //spdlog::info("After New connection {}", port );
  return NULL;
 } else {
	 
  char *msg = theConnection->next_message(); // Leftover from an earlier read?
  if (msg != NULL)
    return msg;

  auto avail = theConnection->available();
  if (avail > 0 ) {
    return( theConnection->read(avail) );
  } else {
    return NULL;
  }
 }
}
