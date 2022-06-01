import streamlit as st
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

st.set_page_config(page_title="Fake News detector", layout="centered")

def main():
	st.title("Fake News Detector")
	st.write("__________________")

	user_input = st.text_input("Enter URL:")

if __name__ == '__main__':
	main()
