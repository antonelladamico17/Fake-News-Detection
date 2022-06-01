import streamlit as st
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

st.set_page_config(page_title="Fake News detector", layout="centered")

def main():
	st.title("Fake Newd Detector")
	st.write("__________________")
	st.text("Enter URL")


	menu = ["Home","About"]
	choice = st.sidebar.selectbox('Menu',menu)
	if choice == 'Home':
		st.subheader("Streamlit From Colab")	



if __name__ == '__main__':
	main()
