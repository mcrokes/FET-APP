import graphviz


try:

    dot = graphviz.Digraph()

    print("Graphviz is installed and in the system's PATH")

except FileNotFoundError:

    print("Graphviz is not installed or not in the system's PATH")