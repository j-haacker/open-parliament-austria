# Contributing to Open Parliament Austria

First off, thank you for considering contributing to Open Parliament Austria! 🎉 

This project aims to make parliamentary data more accessible, and we welcome contributors of all experience levels. Whether you're fixing a typo, improving documentation, or adding new features, your help is appreciated.

## New to Open Source?

Don't worry! We were all beginners once. Here are some resources to help you get started:
- [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/)
- [Understanding Python Basics](https://docs.python.org/3/tutorial/)
- [Git Basics](https://git-scm.com/book/en/v2/Getting-Started-Git-Basics)

## Getting Started

1. **Fork the Repository**
   - Click the "Fork" button at the top right of this page
   - Clone your fork locally: `git clone https://github.com/YOUR-USERNAME/open-parliament-austria.git`

2. **Set Up Your Environment**
   - Install Python 3.11 or newer
   - Create a virtual environment: `python -m venv venv`
   - Activate it: 
     - Windows: `venv\Scripts\activate`
     - Unix/MacOS: `source venv/bin/activate`
   - Install dependencies: `pip install -e ".[dev]"`

3. **Create a Branch**
   - `git checkout -b feature/your-feature-name`
   - or `git checkout -b fix/issue-you-are-fixing`

## Making Changes

1. **Keep it Simple**
   - Start small! Simple changes are easier to review and more likely to be merged quickly
   - One feature or fix per pull request

2. **Follow the Style**
   - We use [Ruff](https://docs.astral.sh/ruff/) for code formatting
   - Add docstrings to functions and classes
   - Add type hints where possible

3. **Test Your Changes**
   - Add tests if you're adding new functionality
   - Run the test suite: `pytest`
   - Ensure all tests pass

## Submitting Changes

1. **Commit Your Changes**
   - Use clear commit messages that explain what and why
   - Example: `git commit -m "Add function to parse parliament sessions"`

2. **Push to Your Fork**
   - `git push origin feature/your-feature-name`

3. **Create a Pull Request**
   - Go to the repository on GitHub
   - Click "New Pull Request"
   - Fill in the template with details about your changes

## Need Help?

- Check out our [Issues](https://github.com/your-org/open-parliament-austria/issues) page
- Look for issues labeled "good first issue" or "help wanted"
- Don't hesitate to ask questions in the issues!

## Code of Conduct

Please note that this project follows our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

Thank you for your contributions! 🙏