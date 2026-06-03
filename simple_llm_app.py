import streamlit as st


st.set_page_config(page_title="Simple LLM App", page_icon="chat")


def simple_llm(user_message: str) -> str:
    message = user_message.strip().lower()
    if message == "hi":
        return "hello"
    return "Please type hi."


st.title("Simple LLM Application")
st.write("Type `hi` and the LLM will reply with `hello`.")

user_message = st.text_input("Your message")

if st.button("Send"):
    response = simple_llm(user_message)
    st.chat_message("user").write(user_message)
    st.chat_message("assistant").write(response)
