#define SOCKET_BUFFER_SIZE 1024 

//Example follows:

#include <boost/asio.hpp>
using boost::asio::ip::tcp;

class Connection {
  boost::asio::io_service ioService;
  std::shared_ptr<boost::asio::ip::tcp::socket> currentSocket;
  bool isConnected;
  char buffer[SOCKET_BUFFER_SIZE];
  int port=0;

public:
  Connection(int port);
  ~Connection();

  int available() {return (int) currentSocket->available(); };
  char* read();
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

char *Connection::read()
{
  boost::system::error_code error;
  size_t length = currentSocket->read_some(boost::asio::buffer(buffer), error);
  if (error == boost::asio::error::eof) {
    spdlog::error("SOCKET {} EOF", port);
  } else if (error) {
    throw boost::system::system_error(error); // Some other error
  }
  return (char *)&buffer;
}

// GLOBAL (in context of includer)
Connection* theConnection=NULL;

// HELPER, uses globals
static char* socket_check(int port) {
if (theConnection==NULL) {
  //spdlog::info("Before New connection {}", port );
  // This blocks until receiver connects:
  theConnection = new Connection(port);
  //spdlog::info("Fater New connection {}", port );
  return NULL;
 } else {
  if (theConnection->available()>0 ) {
    static char *message = theConnection->read();
    spdlog::info("Message @{}: {}", port, message );
    return message;
  } else {
    return NULL;
  }
 }
}
