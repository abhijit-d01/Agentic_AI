from src.states.blogstate import BlogState
import traceback

class BlogNode:
    def __init__(self, llm):
        self.llm = llm

    def title_creation(self, state: BlogState):
        print("\n--- 1. Executing Title Creation ---")
        topic = state.get("topic")
        
        if not topic:
            print("❌ Error: No topic found in state.")
            return {}

        try:
            # Generate Title
            prompt = f"Generate a catchy, SEO-friendly blog title for the topic: '{topic}'. Return ONLY the title."
            response = self.llm.invoke(prompt)
            title = response.content.strip().replace('"', '') # Clean up quotes
            
            print(f"✅ Generated Title: {title}")
            return {"blog": {"title": title}}
            
        except Exception as e:
            print(f"❌ Error in title_creation: {e}")
            traceback.print_exc()
            return {}

    def content_generation(self, state: BlogState):
        print("\n--- 2. Executing Content Generation ---")
        
        # 1. Retrieve inputs safely
        topic = state.get("topic")
        
        # In LangGraph, 'blog' might be None if the previous step failed, so we default to {}
        blog_data = state.get("blog") or {}
        title = blog_data.get("title")

        # Fallback: If title is missing, use the topic as the title
        if not title:
            print("⚠️ Warning: Title not found in state. Using topic as title.")
            title = topic

        try:
            # 2. Generate Content
            print(f"📝 Generating content for title: '{title}'...")
            prompt = (
                f"Act as an expert technical writer. Write a comprehensive blog post about '{topic}'.\n"
                f"The title of the post is: '{title}'.\n"
                f"Use Markdown formatting (headings, bullet points). Write blog in 300 words."
            )
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            print(f"✅ Content Generated! Length: {len(content)} characters.")

            # 3. Return combined dictionary (CRITICAL STEP)
            # We must re-return the title along with the content, 
            # otherwise the 'blog' key gets overwritten with just content.
            return {
                "blog": {
                    "title": title,
                    "content": content
                }
            }

        except Exception as e:
            print(f"❌ Error in content_generation: {e}")
            traceback.print_exc()
            # Return what we have so we don't lose the title in the final state
            return {"blog": {"title": title, "content": f"Error generating content: {str(e)}"}}