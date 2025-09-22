from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
In a small village nestled between misty mountains, there lived a young girl named Elara. She had a curious heart and eyes that sparkled with wonder, always looking beyond the familiar fields and forests of her home.

One autumn morning, as the sun painted the sky in shades of amber and rose, Elara stumbled upon an old, crumbling tower hidden deep in the forest. Its stones were covered with ivy, and the door, though weathered, seemed to hum with quiet energy. Drawn by a mixture of fear and fascination, she pushed it open.

Elara opened the box with trembling hands. Inside lay a delicate golden key, its surface engraved with symbols that shimmered faintly in the sunlight. Alongside it was a note: “This key opens what the heart truly seeks, but beware—the journey is yours alone.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 20,
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)




