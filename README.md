# Knowledge Synthesizer

**Knowledge Synthesizer** is an advanced Python system for extracting, organizing, and synthesizing knowledge from collections of unstructured documents (text, Markdown, CSVs, etc). It builds a multi-level knowledge graph using AI (OpenAI models), then generates readable, structured output in a variety of formats (comprehensive guides, FAQs, checklists, executive summaries, and more).

This is a next-gen knowledge management and synthesis tool: it brings together LLM-powered extraction, graph-based organization, and automated Markdown publishing — giving you detailed, well-organized, and readable documentation from any pile of docs.


---

## Concept: What Does This Script Do?

**Knowledge Synthesizer** transforms loose files into structured, useful knowledge. It does this by:

* Reading all documents in a given directory (the “corpus”)
* Using an LLM (OpenAI) to extract meaningful “knowledge units” (concepts, facts, procedures, etc.)
* Building a multi-level **knowledge graph** connecting all the extracted ideas
* Synthesizing the graph into **structured, readable documentation** — guides, FAQs, checklists, and more
* Saving the output as well-formatted Markdown files

This enables fast creation of knowledge bases, documentation, SOPs, reference manuals, and more — from any pile of legacy or project documents.

---

## How It Works: Conceptual Overview

1. **Corpus Reading**: All supported files in your `corpus/` directory are loaded as source material.

2. **AI Extraction**:

   * Each file is processed in turn.
   * The system uses an OpenAI model to extract “themes” and “knowledge nuggets.”
   * Extracted units are classified by type (conceptual/procedural/etc.) and level (from “pillar” concepts to “atomic” details).

3. **Graph Construction**:

   * All extracted units are organized into a knowledge graph, showing conceptual relationships, dependencies, and hierarchy.
   * Graph integrity is checked and “gaps” in logic can be automatically filled by AI-generated bridging sections.

4. **Synthesis & Publishing**:

   * The user selects how to synthesize the knowledge: comprehensive guide, FAQ, executive summary, best-practices checklist, etc.
   * A Markdown document is generated, with all the structure, style, and clarity you’d expect from a professional technical writer.
   * The output is saved to the output directory, ready for publishing or sharing.

---

## Requirements

* **Python 3.8 or newer**

* **Python Packages:**
  Before running the script, make sure you have these packages installed:

  ```
  pip install openai backoff
  ```

  * `openai`: Python client for the OpenAI API
  * `backoff`: Handles automatic retries for API calls

---

## Setting Your OpenAI API Key

This script requires access to OpenAI’s API. **You must set your API key in the environment before running the script.**

**On Windows:**

```cmd
set OPENAI_API_KEY=sk-xxxxxx...
python ks.py
```

**On Mac/Linux:**

```bash
export OPENAI_API_KEY=sk-xxxxxx...
python ks.py
```

* Get your key from [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
* The key should start with `sk-` or `sk-proj-` and should be kept secure.
* If you want this to be automatic, add the export command to your `.bashrc` or `.zshrc`, or use Windows environment variable settings.

The script looks for `OPENAI_API_KEY` automatically.

---

## Usage: Step-by-Step Instructions

### 1. Place Files in the Corpus Directory

* Put all source files you want to synthesize into the `corpus/` directory in your repo/project.
* Files can be `.txt`, `.md`, `.csv`, `.json`, `.yaml`, `.yml`, `.rst`, `.tsv`, `.toml`, `.ini`, or `.log`.

### 2. Supported File Types

* Text files: `.txt`, `.md`, `.rst`
* Data/config: `.csv`, `.tsv`, `.json`, `.yaml`, `.yml`, `.toml`, `.ini`, `.log`
* All must be UTF-8 encoded and not too large (the script will chunk large files if needed).

### 3. Running the Script

From your project directory, run:

```bash
python ks.py
```

You’ll see an interactive menu guiding you through the options.

### 4. Interactive Options & What They Mean

As you run the script, you’ll be prompted to select:

* **Writing Style**: Choose the style of the output document.

  * Examples: “Upbeat and energetic”, “Concise and to-the-point”, “Formal and academic”, or enter your own custom style.

* **Output Format**: Choose what kind of document you want:

  * `comprehensive_guide`: Classic all-in-one documentation/manual
  * `quick_reference`: Markdown table cheat sheet
  * `faq`: Grouped Q\&A format
  * `step_by_step_manual`: Instructions/workflows
  * `executive_summary`: Short, high-level briefing
  * `annotated_outline`: Structured outline with explanations
  * `best_practices_checklist`: Actionable checklist
  * `case_studies`, `problem_solution_playbook`, `visual_roadmap`: Specialized outputs (extendable)

* **Structural Mode**:

  * `flat`: Content is organized by topic/level, like a textbook.
  * `structural`: Output is shaped by the conceptual dependencies in the knowledge graph (more advanced, “how things fit together”).

* **Gap Filling**: Optionally, the AI can identify and “fill in” logical gaps in the document with new explanatory sections.

* **Regeneration**: After one run, you can immediately regenerate with different options, or rebuild the knowledge graph if you change your corpus.

### 5. Output Files

* Output Markdown files are saved in the configured output directory (default is the project root).
* Filenames are auto-generated based on the content, format, and time (e.g., `Core_Concepts_Quick_Reference_20250712_083455.md`).
* Each run prints the output path and document to the screen.

---

## Examples Directory

The repository includes an `examples/` directory to help you understand and test Knowledge Synthesizer. Here you’ll find:

* **Static\_electricity\_(original\_content).txt**
  The raw, original source text about static electricity. This file demonstrates the kind of input you can provide to the system.

* **Fundamentals and Experiments in Frictional Electricity.pdf**
  An example of an auto-generated comprehensive guide produced by the system from the above source.

* **Frequently Asked Questions (FAQ).pdf**
  An example of an auto-generated FAQ-style output, using the same knowledge graph.

These examples are meant to show what kinds of input the tool works with, and what the output looks like in different synthesis modes.

---
