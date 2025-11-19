// capture_code_snapshot.js (v6.0) - Smart-Secrets-Scanner Code Packaging Tool
//
// This script packages all relevant code and documentation from the Smart-Secrets-Scanner
// project into a single shareable file for LLM review and analysis.
//
// Usage: node capture_code_snapshot.js [output_file]
//
// The script will:
// 1. Scan the project directory for relevant files
// 2. Include source code, documentation, and configuration files
// 3. Exclude large binary files, logs, and model artifacts
// 4. Package everything into a single text file with clear file separators

const fs = require('fs');
const path = require('path');

// Parse command line arguments
const outputFile = process.argv[2] || 'smart_secrets_scanner_snapshot.txt';

// Project root directory
const projectRoot = __dirname;

// Files and directories to always exclude
const alwaysExcludeFiles = [
    // Git repository files (exclude everything in .git/)
    '.git/',
    // Large model files and checkpoints
    'models/',
    'outputs/checkpoints/',
    'models/merged/',
    'outputs/logs/',
    'reference-files/',
    // Data directories (can be very large)
    'data/evaluation/processed/',
    'data/raw/',
    // Project management files (not needed for code review)
    'tasks/',
    // Git and system files
    '.gitignore',
    'node_modules/',
    '.DS_Store',
    // Temporary and cache files
    '*.tmp',
    '*.log',
    '*.cache',
    // Python cache
    '__pycache__/',
    '*.pyc',
    '*.pyo',
    // Environment files
    '.env',
    'venv/',
    'env/',
    // Build artifacts
    'dist/',
    'build/',
    // IDE files
    '.vscode/',
    '.idea/',
    // OS files
    'Thumbs.db',
    'Desktop.ini',
    // Output files
    'smart_secrets_scanner_snapshot.txt',
    // GGUF model files (large binary files)
    '**/*.gguf'
];

// Allowed file extensions for content inclusion
const allowedExtensions = new Set([
    '.md',      // Documentation
    '.py',      // Python source code
    '.js',      // JavaScript
    '.json',    // Configuration files
    '.yaml',    // YAML configuration
    '.yml',     // YAML configuration
    '.toml',    // TOML configuration
    '.sh',      // Shell scripts
    '.ps1',     // PowerShell scripts
    '.txt',     // Text files
    '.cfg',     // Configuration files
    '.ini'      // Configuration files
]);

// File separators for the output
const fileSeparatorStart = '\n\n' + '='.repeat(80) + '\n--- START OF FILE: ';
const fileSeparatorEnd = '\n--- END OF FILE: ';
const fileSeparatorDivider = '\n' + '='.repeat(80) + '\n\n';

/**
 * Check if a file should be excluded based on patterns
 */
function shouldExcludeFile(filePath) {
    const relativePath = path.relative(projectRoot, filePath).replace(/\\/g, '/'); // Normalize to forward slashes

    // Allow specific files even if they're in excluded directories
    const allowedFiles = [
        'models/README.md'
    ];

    if (allowedFiles.includes(relativePath)) {
        return false;
    }

    return alwaysExcludeFiles.some(pattern => {
        const normalizedPattern = pattern.replace(/\\/g, '/'); // Normalize pattern too
        if (normalizedPattern.includes('/')) {
            // Directory pattern
            return relativePath.startsWith(normalizedPattern) || relativePath.includes('/' + normalizedPattern);
        } else if (normalizedPattern.startsWith('*.')) {
            // File extension pattern
            return relativePath.endsWith(normalizedPattern.substring(1));
        } else {
            // Exact file match
            return relativePath === normalizedPattern || relativePath.endsWith('/' + normalizedPattern);
        }
    });
}

/**
 * Check if a file extension is allowed
 */
function isAllowedExtension(filePath) {
    const ext = path.extname(filePath).toLowerCase();
    return allowedExtensions.has(ext) || ext === ''; // Include files without extensions
}

/**
 * Recursively get all files in a directory
 */
function getAllFiles(dirPath, fileList = []) {
    try {
        const files = fs.readdirSync(dirPath);

        for (const file of files) {
            const filePath = path.join(dirPath, file);
            const stat = fs.statSync(filePath);

            if (stat.isDirectory()) {
                // Skip excluded directories
                if (!shouldExcludeFile(filePath + '/')) {
                    getAllFiles(filePath, fileList);
                }
            } else {
                // Include file if not excluded and has allowed extension
                if (!shouldExcludeFile(filePath) && isAllowedExtension(filePath)) {
                    fileList.push(filePath);
                }
            }
        }
    } catch (error) {
        console.warn(`Warning: Could not read directory ${dirPath}: ${error.message}`);
    }

    return fileList;
}

/**
 * Read file content safely
 */
function readFileContent(filePath) {
    try {
        // Check file size (skip files larger than 10MB)
        const stat = fs.statSync(filePath);
        if (stat.size > 10 * 1024 * 1024) {
            return `[FILE TOO LARGE - ${stat.size} bytes]`;
        }

        return fs.readFileSync(filePath, 'utf8');
    } catch (error) {
        return `[ERROR READING FILE: ${error.message}]`;
    }
}

/**
 * Generate the code snapshot
 */
function generateSnapshot() {
    console.log('🔍 Scanning Smart-Secrets-Scanner project files...');

    const allFiles = getAllFiles(projectRoot);
    console.log(`📁 Found ${allFiles.length} files to process`);

    let snapshot = '';

    // Add header
    snapshot += '='.repeat(80) + '\n';
    snapshot += 'SMART-SECRETS-SCANNER CODE SNAPSHOT\n';
    snapshot += 'Generated on: ' + new Date().toISOString() + '\n';
    snapshot += 'Project: Smart-Secrets-Scanner (Llama 3.1 Fine-tuning for Secret Detection)\n';
    snapshot += '='.repeat(80) + '\n\n';

    snapshot += 'PROJECT OVERVIEW:\n';
    snapshot += 'This snapshot contains all relevant source code, documentation, and configuration\n';
    snapshot += 'files from the Smart-Secrets-Scanner project. The project focuses on fine-tuning\n';
    snapshot += 'Llama 3.1 models to detect secrets and sensitive information in code.\n\n';

    snapshot += 'KEY COMPONENTS:\n';
    snapshot += '- CUDA-enabled PyTorch environment setup\n';
    snapshot += '- Model fine-tuning scripts\n';
    snapshot += '- Inference and evaluation tools\n';
    snapshot += '- Dataset validation and processing\n';
    snapshot += '- Model deployment with Ollama\n\n';

    snapshot += 'FILES INCLUDED:\n';
    allFiles.forEach(file => {
        const relativePath = path.relative(projectRoot, file);
        snapshot += `- ${relativePath}\n`;
    });
    snapshot += '\n' + '='.repeat(80) + '\n\n';

    // Process each file
    let processedCount = 0;
    for (const filePath of allFiles) {
        processedCount++;
        const relativePath = path.relative(projectRoot, filePath);
        const fileName = path.basename(filePath);

        console.log(`📄 Processing ${processedCount}/${allFiles.length}: ${relativePath}`);

        const content = readFileContent(filePath);

        snapshot += fileSeparatorStart + relativePath + fileSeparatorDivider;
        snapshot += content;
        snapshot += fileSeparatorEnd + relativePath + '\n';
        snapshot += fileSeparatorDivider;
    }

    // Add footer
    snapshot += '='.repeat(80) + '\n';
    snapshot += 'END OF SMART-SECRETS-SCANNER CODE SNAPSHOT\n';
    snapshot += 'Total files processed: ' + allFiles.length + '\n';
    snapshot += '='.repeat(80) + '\n';

    return snapshot;
}

/**
 * Main execution
 */
function main() {
    try {
        console.log('🚀 Starting Smart-Secrets-Scanner code snapshot generation...');

        const snapshot = generateSnapshot();

        // Write to output file
        fs.writeFileSync(outputFile, snapshot, 'utf8');

        console.log(`✅ Snapshot generated successfully: ${outputFile}`);
        console.log(`📊 File size: ${(fs.statSync(outputFile).size / 1024 / 1024).toFixed(2)} MB`);

    } catch (error) {
        console.error('❌ Error generating snapshot:', error.message);
        process.exit(1);
    }
}

// Run the script
if (require.main === module) {
    main();
}

module.exports = { generateSnapshot, getAllFiles, shouldExcludeFile };