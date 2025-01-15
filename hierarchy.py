from dataclasses import dataclass

class Paper:
    def __init__(self, paper_id, title=None, abstract=None, segments=None):
        self.paper_id = paper_id
        self.title = title
        self.abstract = abstract
        self.segments: list[Segment] = segments
        
    def get_summary(self, max_length=250):
        """Returns a summary of the paper."""
        return {
            "Title": self.title,
            "Abstract": self.abstract[:max_length] + "..." if len(self.abstract) > max_length else self.abstract,
        }
    
@dataclass
class Segment:
    global_id: int
    local_id: int
    paper_id: int
    content: str

class AspectNode:
    def __init__(self, idx, name, parent=None, keywords=None):
        self.id = idx
        self.name = name
        self.keywords = keywords
        self.parent = parent
        self.depth = 0 if self.parent is None else self.parent.depth + 1

        self.sub_aspects = []
        self.ranked_segments = []
        self.related_papers = {}  # Dictionary for faster lookup by paper_id

    def add_sub_aspect(self, sub_aspect):
        """Adds a sub-aspect to the current node."""
        self.sub_aspects.append(sub_aspect)

    def add_related_paper(self, paper, relevant_segments):
        """Links a paper and its associated relevant segments to this aspect node."""
        if paper.paper_id in self.related_papers:
            curr_segments = self.related_papers[paper.paper_id]["relevant_segments"]
            for seg in relevant_segments:
                if seg not in curr_segments:
                    curr_segments.append(seg)
            self.related_papers[paper.paper_id]["relevant_segments"] = curr_segments
        else:
            self.related_papers[paper.paper_id] = {
                "paper": paper,
                "relevant_segments": relevant_segments,
                "perspective_statement": None  # Initially None, to be updated later
            }

    def update_perspective_statement(self, paper_id, perspective_statement):
        """Updates the perspective statement for a specific paper."""
        if paper_id in self.related_papers:
            self.related_papers[paper_id]["perspective_statement"] = perspective_statement
        else:
            raise ValueError(f"Paper with ID '{paper_id}' not found in related papers.")
    
    def get_parent(self):
        return self.parent
    
    def get_siblings(self):
        return [aspect_node for aspect_node in self.parent.sub_aspects if aspect_node.id != self.id]

    def get_all_papers(self):
        """Retrieves all papers associated with this node and its sub-aspects."""
        papers = set(entry["paper"] for entry in self.related_papers.values())
        for sub_aspect in self.sub_aspects:
            papers.update(sub_aspect.get_all_papers())
        return papers

    def get_all_keywords(self):
        """Retrieves all keywords associated with this node and its sub-aspects."""
        keywords = set(self.keywords)
        for sub_aspect in self.sub_aspects:
            keywords.update(sub_aspect.get_all_keywords())
        return keywords

    def get_all_segments(self, as_str=False):
        """Retrieves all segments associated with this node and its papers."""
        all_segments = []
        for entry in self.related_papers.values():
            if as_str:
                all_segments.extend([seg.content for seg in entry["relevant_segments"]])
            else:
                all_segments.extend(entry["relevant_segments"])
        
        return all_segments
    
    def display(self, indent_multiplier=2, visited=None):
        """
        Displays the aspect hierarchy starting from this node.

        Args:
        indent_multiplier (int): The number of spaces used for indentation, multiplied by the depth.
        visited (set): A set of visited node IDs to handle cycles in the directed acyclic graph.
        """
        if visited is None:
            visited = set()
        if self.id in visited:
            return
        visited.add(self.id)

        indent = " " * (self.depth * indent_multiplier)
        print(f"{indent}Aspect: {self.name}")
        print(f"{indent}Depth: {self.depth}")
        print(f"{indent}Keywords: {self.keywords}")
        print(f"{indent}Top #1 Segment ({self.ranked_segments[0][1]}): {self.ranked_segments[0][0].content}")
        print(f"{indent}Top #2 Segment ({self.ranked_segments[1][1]}): {self.ranked_segments[1][0].content}")
        print(f"{indent}Top #3 Segment ({self.ranked_segments[2][1]}): {self.ranked_segments[2][0].content}")
        if self.sub_aspects:
            print(f"{indent}{'-'*40}")
            print(f"{indent}Subaspects:")
            for subaspect in self.sub_aspects:
                subaspect.display(self.depth + 1, indent_multiplier, visited)
        print(f"{indent}{'-'*40}")


class Tree:
    def __init__(self, root):
        self.root = root

    def add_aspect(self, parent_node, aspect_node):
        """Adds an aspect node to the tree under the specified parent node."""
        parent_node = self.find_node(self.root, parent_node)
        if parent_node:
            parent_node.add_sub_aspect(aspect_node)
        else:
            raise ValueError(f"Parent node '{parent_node.name}' not found in the tree.")

    def find_node(self, current_node, target_node):
        """Helper function to find a node by name."""
        if current_node.id == target_node.id:
            return current_node
        for sub_aspect in current_node.sub_aspects:
            result = self.find_node(sub_aspect, target_node)
            if result:
                return result
        return None

    def display_tree(self):
        """Displays the entire aspect hierarchy."""
        self.root.display()