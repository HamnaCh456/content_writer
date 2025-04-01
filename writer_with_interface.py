import os
import re
import sys
import time
import streamlit as st
from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from crewai import Crew, Agent, Task, Process, LLM
from crewai.tools import tool

# Global variables
task_values = []

# Define the StreamToExpander class for displaying output in Streamlit
class StreamToExpander:
    def __init__(self, expander):
        self.expander = expander
        self.buffer = []
        self.colors = ['blue', 'green', 'purple', 'teal']  # Changed colors to avoid red
        self.color_index = 0  # Initialize color index

    def write(self, data):
        # Filter out ANSI escape codes using a regular expression
        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)

        # Check if the data contains 'task' information
        task_match_object = re.search(r'\"task\"\s*:\s*\"(.*?)\"', cleaned_data, re.IGNORECASE)
        task_match_input = re.search(r'task\s*:\s*([^\n]*)', cleaned_data, re.IGNORECASE)
        task_value = None
        if task_match_object:
            task_value = task_match_object.group(1)
        elif task_match_input:
            task_value = task_match_input.group(1).strip()

        if task_value and task_value not in task_values:
            task_values.append(task_value)
            st.toast(":robot_face: " + task_value)

        # Check if the text contains the specified phrase and apply color
        if "Entering new CrewAgentExecutor chain" in cleaned_data:
            # Apply different color and switch color index
            self.color_index = (self.color_index + 1) % len(self.colors)  # Increment color index and wrap around if necessary
            cleaned_data = cleaned_data.replace("Entering new CrewAgentExecutor chain", f":{self.colors[self.color_index]}[Entering new CrewAgentExecutor chain]")

        # Highlight different agent roles with colors
        if "Content Planner" in cleaned_data:
            cleaned_data = cleaned_data.replace("Content Planner", f":{self.colors[self.color_index]}[Content Planner]")
        if "Content Writer" in cleaned_data:
            cleaned_data = cleaned_data.replace("Content Writer", f":{self.colors[self.color_index]}[Content Writer]")
        if "Finished chain." in cleaned_data:
            cleaned_data = cleaned_data.replace("Finished chain.", f":{self.colors[self.color_index]}[Finished chain.]")

        self.buffer.append(cleaned_data)
        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer = []

# Function to create and run the CrewAI setup
def create_crewai_setup(topic):
    # Set API key - you can replace this with your own API key or environment variable
    os.environ["GROQ_API_KEY"] = "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"
    
    # Initialize LLM
    llm = LLM(model="groq/llama-3.3-70b-specdec")
    
    @tool("DUCKDUCKGoSEARCH")
    def my_simple_tool(query: str) -> str:
        """THIS TOOL SEARCH ON THE WEB FOR YOU TAKING A QUERY AS AN INPUT"""
        # Tool logic here
        duckduckgo_search = DuckDuckGoSearchRun()
        result = duckduckgo_search.invoke(query)
        return result

    # Define the Content Planner agent
    planner = Agent(
        role="Content Planner",
        goal=f"Plan engaging and factually accurate content on {topic}",
        backstory=f"You're working on planning a blog article "
                f"about the topic: {topic}."
                f"You collect information that helps the "
                f"audience learn something "
                f"and make informed decisions. "
                f"Your work is the basis for "
                f"the Content Writer to write an article on this topic.",
        llm=llm,        
        allow_delegation=False,
        tools=[my_simple_tool],
        verbose=True
    )

    # Define the Content Writer agent
    writer = Agent(
        role="Content Writer",
        goal=f"Write insightful and factually accurate "
            f"opinion piece about the topic: {topic}",
        backstory=f"You're working on a writing "
                f"a new opinion piece about the topic: {topic}. "
                f"You base your writing on the work of "
                f"the Content Planner, who provides an outline "
                f"and relevant context about the topic. "
                f"You follow the main objectives and "
                f"direction of the outline, "
                f"as provide by the Content Planner. "
                f"You also provide objective and impartial insights "
                f"and back them up with information "
                f"provide by the Content Planner. "
                f"You acknowledge in your opinion piece "
                f"when your statements are opinions "
                f"as opposed to objective statements.",
        llm=llm,
        allow_delegation=False,
        verbose=True
    )

    # Define the planning task
    plan = Task(
        description=(
            f"1. Prioritize the latest trends, key players, "
                f"and noteworthy news on {topic}.\n"
            f"2. Identify the target audience, considering "
                f"their interests and pain points.\n"
            f"3. Develop a detailed content outline including "
                f"an introduction, key points, and a call to action.\n"
            f"4. Include SEO keywords and relevant data or sources."
        ),
        expected_output="A comprehensive content plan document "
            "with an outline, audience analysis, "
            "SEO keywords, and resources, maximum of 2 or 3 paragraph length ",
        agent=planner,
    )
    
    # Define the writing task
    write = Task(
        description=(
            f"1. Use the content plan to craft a compelling "
                f"blog post on {topic}.\n"
            f"2. Incorporate SEO keywords naturally.\n"
            f"3. Sections/Subtitles are properly named "
                f"in an engaging manner.\n"
            f"4. Ensure the post is structured with an "
                f"engaging introduction, insightful body, "
                f"and a summarizing conclusion.\n"
            f"5. Proofread for grammatical errors and "
                f"alignment with the brand's voice.\n"
        ),
        expected_output="A well-written blog post "
            "in markdown format, ready for publication, "
            "each section should have 2 or 3 paragraphs.",
        agent=writer,
    )

    # Create and run the crew
    crew = Crew(
        agents=[planner, writer],
        tasks=[plan, write],
        process=Process.sequential,
        verbose=True,
        telemetry=False
    )
    
    # Execute the crew with the topic
    result = crew.kickoff(inputs={"topic": topic})
    return result

# Streamlit interface
def run_crewai_writer_app():
    st.title("AI Content Creation Crew")
    
    # Add toggle for thought process visibility
    show_thought_process = st.sidebar.checkbox("Show AI Thought Process", value=False)
    
    if not show_thought_process:
        with st.expander("About the Team:", expanded=True):
            st.subheader("Content Creation Process")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Content Planner")
                st.markdown("""
                **Role**: Content Planner  
                **Goal**: Plan engaging and factually accurate content on a given topic  
                **Backstory**: Expert at planning blog articles that inform and engage the target audience.  
                Collects relevant information to help the audience learn something new and make informed decisions.  
                **Task**: Prioritize trends, identify target audience, develop content outline, and include SEO keywords.
                """)
            
            with col2:
                st.subheader("Content Writer")
                st.markdown("""
                **Role**: Content Writer  
                **Goal**: Write insightful and factually accurate opinion pieces  
                **Backstory**: Skilled writer who creates compelling content based on the Content Planner's outline.  
                Provides objective insights backed by information from the planner.  
                **Task**: Craft engaging blog posts with proper structure, SEO keywords, and clear sections.
                """)
    
    topic = st.text_input("Enter a topic for your blog post:", placeholder="e.g., Artificial Intelligence, Climate Change, Digital Marketing")
    
    if st.button("Generate Content"):
        if not topic:
            st.error("Please enter a topic before generating content.")
            return
        
        # Clear task values for new run
        global task_values
        task_values = []
        
        # Placeholder for stopwatch
        stopwatch_placeholder = st.empty()
        
        # Start the stopwatch
        start_time = time.time()
        
        # Only show thought process if the toggle is enabled
        if show_thought_process:
            with st.expander("Processing - Watch the agents in action!", expanded=True):
                # Redirect stdout to our StreamToExpander
                sys.stdout = StreamToExpander(st)
                with st.spinner("Generating Content..."):
                    crew_result = create_crewai_setup(topic)
                # Reset stdout to its default value
                sys.stdout = sys.__stdout__
        else:
            # Just show a spinner without the thought process
            with st.spinner("Generating Content..."):
                # Redirect stdout to devnull to hide the process
                original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                crew_result = create_crewai_setup(topic)
                # Reset stdout
                sys.stdout.close()
                sys.stdout = original_stdout
        
        # Stop the stopwatch
        end_time = time.time()
        total_time = end_time - start_time
        stopwatch_placeholder.text(f"Total Time Elapsed: {total_time:.2f} seconds")
        
        if task_values and show_thought_process:
            st.header("Tasks Completed:")
            st.table({"Tasks": task_values})
        
        st.header("Final Blog Post:")
        
        # Extract the actual content string from CrewOutput object
        result_content = str(crew_result)
        
        st.markdown(result_content)
        
        # Convert the string to bytes for download button
        result_bytes = result_content.encode('utf-8')
        
        # Add download button for the result
        st.download_button(
            label="Download Blog Post",
            data=result_bytes,
            file_name=f"{topic.replace(' ', '_')}_blog_post.md",
            mime="text/markdown"
        )

# Run the app if this file is executed directly
if __name__ == "__main__":
    run_crewai_writer_app()