# Answer: Why MSI and ZIP Package Sizes Differ

## Quick Answer

**MSI packages are typically 10-30% larger than ZIP packages** because:

1. **MSI includes installer infrastructure** (scripts, UI, metadata) - adds 2-5MB
2. **System integration data** (registry operations, uninstall info) - adds 1-3MB  
3. **Reliability features** (checksums, rollback info, signatures) - adds 5-10%
4. **Different compression** - MSI uses CAB (less efficient than modern ZIP)

## Size Example

For a 100MB application:
- **ZIP**: ~50MB (just compressed files)
- **MSI**: ~60-65MB (files + installer overhead)

## About This Repository

**Important**: This repository (SwingAgent) is a **Python package** that uses **pip/wheel** for installation, NOT MSI or ZIP packages.

```bash
# SwingAgent installation (Python way)
pip install -e .
```

For detailed information:
- See [docs/packaging-formats.md](docs/packaging-formats.md) for complete MSI vs ZIP explanation
- See [README.md](README.md) for SwingAgent installation instructions
- See [docs/faq.md](docs/faq.md) for common questions

## Summary

**Why MSI is Larger:**
- Contains installation wizard and system integration
- Includes metadata for Windows Add/Remove Programs
- Has rollback and repair capabilities
- Uses less efficient compression

**Why ZIP is Smaller:**
- Just compressed files, no installer
- No system integration overhead
- More efficient compression possible

This is a general Windows packaging concept and does not apply to Python packages like SwingAgent.
