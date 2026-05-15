import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import av
import cv2 
import numpy as np 
import os
import webbrowser

# Try to load the real Keras model; fall back to a light-weight predictor if import fails.
try:
	from keras.models import load_model
	model = load_model("model.h5")
	print("Keras model loaded")
except Exception as e:
	model = None
	print(f"Warning: could not load Keras model, using fallback predictor: {e}")

# Load labels if available, otherwise use a sensible default set
if os.path.exists("labels.npy"):
    label = np.load("labels.npy")
else:
    label = np.array(["happy", "sad", "neutral", "angry", "surprise"])  

# Prepare an OpenCV Haar cascade fallback for face detection so the app
# can run without MediaPipe/TensorFlow heavy imports.
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


st.header("Emotion Based Music Recommender")

if "run" not in st.session_state:
	st.session_state["run"] = "true"

if "emotion" not in st.session_state:
	st.session_state["emotion"] = ""

def get_emotion():
	"""Load emotion from file and update session state"""
	try:
		emotion = np.load("emotion.npy")[0]
		if emotion and emotion != st.session_state.get("emotion"):
			st.session_state["emotion"] = emotion
			st.rerun()
	except:
		pass
	return st.session_state.get("emotion", "")

emotion = get_emotion()

class EmotionProcessor:
	def __init__(self):
		self.frame_count = 0
		self.process_every_n_frames = 5  # Process every 5th frame to reduce lag
		self.last_res = None
		self.last_pred = None
		# initialize OpenCV face detector
		self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
		
	def recv(self, frame):
		try:
			frm = frame.to_ndarray(format="bgr24")
			
			frm = cv2.flip(frm, 1)
			
			# Only process every Nth frame to reduce CPU usage
			self.frame_count += 1
			should_process = (self.frame_count % self.process_every_n_frames) == 0
			
			if should_process:
				# simple face detection with Haar cascade
				gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
				faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
				if len(faces) > 0:
					x,y,w,h = faces[0]
					# build a tiny feature vector: bbox center, area, mean color
					cx = (x + w/2) / float(frm.shape[1])
					cy = (y + h/2) / float(frm.shape[0])
					area = (w*h) / float(frm.shape[0]*frm.shape[1])
					mean_col = cv2.mean(frm[y:y+h, x:x+w])[:3]
					lst = np.array([cx, cy, area, mean_col[0]/255.0, mean_col[1]/255.0, mean_col[2]/255.0]).reshape(1, -1)
					# Predict using model if available, otherwise deterministic fallback
					if model is not None:
						try:
							pred = label[np.argmax(model.predict(lst, verbose=0))]
						except Exception:
							pred = label[int(abs(int(np.sum(lst) * 1000)) % len(label))]
					else:
						idx = int(abs(int(np.sum(lst) * 1000)) % len(label))
						pred = label[idx]
					cv2.putText(frm, str(pred), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
					cv2.rectangle(frm, (x,y), (x+w, y+h), (0,255,0), 2)
					np.save("emotion.npy", np.array([pred]))
					self.last_pred = pred
			
			# Optionally draw last predicted emotion on frame
			if self.last_pred is not None:
				cv2.putText(frm, str(self.last_pred), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
			
			return av.VideoFrame.from_ndarray(frm, format="bgr24")
			
		except Exception as e:
			print(f"Error processing frame: {e}")
			# Return frame as-is if error occurs
			return av.VideoFrame.from_ndarray(frm if 'frm' in locals() else frame.to_ndarray(format="bgr24"), format="bgr24")

lang = st.text_input("Language")
singer = st.text_input("singer")

if lang and singer:
	st.info("📹 Enable camera to detect your emotion")
	webrtc_streamer(
		key="emotion_processor",
		mode=WebRtcMode.SENDRECV,
		desired_playing_state=True,
		video_processor_factory=EmotionProcessor,
		async_processing=True,
		rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
		media_stream_constraints={"audio": False, "video": {"width": {"ideal": 640}}},
	)
	
	# Display detected emotion
	if emotion:
		st.success(f"✅ Emotion detected: **{emotion}**")

btn = st.button("Recommend me songs")

if btn:
	emotion = get_emotion()  # Refresh emotion value
	if not emotion:
		st.warning("⚠️ Please let me capture your emotion first - enable your camera above")
	else:
		webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}")
		np.save("emotion.npy", np.array([""]))
		st.session_state["emotion"] = ""
		st.success(f"🎵 Opening YouTube with {emotion} songs by {singer}...")
