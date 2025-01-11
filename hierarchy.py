class Paper:
    def __init__(self, paper_id, title, abstract):
        self.paper_id = paper_id
        self.title = title
        self.abstract = abstract
        
    def get_summary(self, max_length=250):
        """Returns a summary of the paper."""
        return {
            "Title": self.title,
            "Abstract": self.abstract[:max_length] + "..." if len(self.abstract) > max_length else self.abstract,
        }

class AspectNode:
    def __init__(self, idx, name, keywords=None):
        self.id = idx
        self.name = name
        self.keywords = keywords
        self.sub_aspects = []
        self.related_papers = {}  # Dictionary for faster lookup by paper_id

    def add_sub_aspect(self, sub_aspect):
        """Adds a sub-aspect to the current node."""
        self.sub_aspects.append(sub_aspect)

    def add_related_paper(self, paper, relevant_segments):
        """Links a paper and its associated relevant segments to this aspect node."""
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

    def get_all_papers(self):
        """Retrieves all papers associated with this node and its sub-aspects."""
        papers = set(entry["paper"] for entry in self.related_papers.values())
        for sub_aspect in self.sub_aspects:
            papers.update(sub_aspect.get_all_papers())
        return papers

    def display(self, level=0):
        """Displays the aspect hierarchy starting from this node."""
        indent = "  " * level
        print(f"{indent}- {self.name}")
        for sub_aspect in self.sub_aspects:
            sub_aspect.display(level + 1)
        if self.related_papers:
            print(f"{indent}  Related Papers:")
            for entry in self.related_papers.values():
                perspective = entry['perspective_statement'] or "[Perspective not set]"
                print(f"{indent}    - {entry['paper'].title}: {perspective}")


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