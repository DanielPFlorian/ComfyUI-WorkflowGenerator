# GitHub Wiki Setup Guide

This directory contains the wiki pages for the ComfyUI-WorkflowGenerator GitHub Wiki.

## How to Upload to GitHub Wiki

GitHub Wikis are stored in a separate repository. Here's how to set it up:

### Method 1: Using GitHub Web Interface (Easiest)

1. Go to your repository: `https://github.com/danielpflorian/ComfyUI-WorkflowGenerator`
2. Click on the **"Wiki"** tab
3. Click **"Create the first page"** or **"New Page"**
4. Copy the content from each `.md` file in this directory
5. Paste into the GitHub Wiki editor
6. Use the filename (without `.md`) as the page title
7. Save the page

### Method 2: Using Git (Recommended for Bulk Upload)

1. **Clone the wiki repository:**
   ```bash
   git clone https://github.com/danielpflorian/ComfyUI-WorkflowGenerator.wiki.git
   ```

2. **Copy the wiki files:**
   ```bash
   cp wiki/*.md ComfyUI-WorkflowGenerator.wiki/
   ```

3. **Commit and push:**
   ```bash
   cd ComfyUI-WorkflowGenerator.wiki
   git add *.md
   git commit -m "Add wiki pages"
   git push origin master
   ```

## Page Structure

The wiki is organized as follows:

- **Home.md** - Main landing page with navigation
- **Installation.md** - Detailed installation instructions
- **Node-Reference.md** - Complete node documentation
- **Configuration.md** - Configuration options and performance tuning
- **Advanced-Usage.md** - Advanced patterns and techniques
- **Troubleshooting.md** - Common issues and solutions
- **Instruction-Prompt-Usage.md** - How instructions work across pipeline steps

## Linking Between Pages

Wiki pages use simple markdown links without file extensions:

```markdown
[Link Text](Page-Name)
```

For example:
- `[Home](Home)` links to the Home page
- `[Installation](Installation)` links to the Installation page
- `[Node Reference](Node-Reference)` links to the Node-Reference page

## Updating the README

After uploading the wiki, update the main repository README to link to the wiki:

```markdown
For detailed documentation, see the [Wiki](https://github.com/danielpflorian/ComfyUI-WorkflowGenerator/wiki).
```

## Notes

- GitHub Wikis automatically create a sidebar with all pages
- The first page you create becomes the home page (or you can name it `Home.md`)
- Page names are case-sensitive and spaces should be replaced with hyphens
- You can add a custom sidebar by creating a `_Sidebar.md` file

