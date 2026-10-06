# Header stamp is image capture time, frame_id is the camera optical frame.
std_msgs/Header header
string camera
string kind
float64 u
float64 v
float64 confidence
geometry_msgs/Point32[] corners
string payload
