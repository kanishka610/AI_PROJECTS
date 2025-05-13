import streamlit as st

# Define simple responses
def chatbot_response(user_input):
    user_input = user_input.lower()
    if "headache" in user_input:
        return "You might be experiencing a migraine. Stay hydrated and take rest."
    elif "fever" in user_input:
        return "Check your temperature and consult a doctor if it is above 100°F."
    elif "appointment" in user_input:
        return "You can book an appointment at our nearest clinic through our portal."
    elif "cold" in user_input:
        return "Drink warm fluids and rest. If it persists, consult a doctor."
    elif "thanks" in user_input or "thank you" in user_input:
        return "You're welcome! Stay healthy. 😊"
    else:
        return "Sorry, I’m not trained to answer that yet. Please contact our support."

# Streamlit UI
st.title("🏥 Healthcare Chatbot")
st.write("Ask health-related questions like symptoms, appointments, or general advice.")

# Input box
user_input = st.text_input("You:", "")

# Show response
if user_input:
    response = chatbot_response(user_input)
    st.markdown(f"**Bot:** {response}")
