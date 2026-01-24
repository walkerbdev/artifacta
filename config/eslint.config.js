import js from '@eslint/js';
import react from 'eslint-plugin-react';
import reactHooks from 'eslint-plugin-react-hooks';
import jsdoc from 'eslint-plugin-jsdoc';

export default [
  js.configs.recommended,
  jsdoc.configs['flat/recommended'],
  {
    files: ['src/**/*.{js,jsx}'],
    plugins: {
      react,
      'react-hooks': reactHooks,
      jsdoc,
    },
    languageOptions: {
      parserOptions: {
        ecmaVersion: 'latest',
        sourceType: 'module',
        ecmaFeatures: {
          jsx: true,
        },
      },
      globals: {
        // Browser globals
        window: 'readonly',
        document: 'readonly',
        navigator: 'readonly',
        console: 'readonly',
        setTimeout: 'readonly',
        clearTimeout: 'readonly',
        setInterval: 'readonly',
        clearInterval: 'readonly',
        fetch: 'readonly',
        FormData: 'readonly',
        Blob: 'readonly',
        URL: 'readonly',
        File: 'readonly',
        FileReader: 'readonly',
        localStorage: 'readonly',
        WebSocket: 'readonly',
        Event: 'readonly',
        CustomEvent: 'readonly',
        URLSearchParams: 'readonly',
        MutationObserver: 'readonly',
        IntersectionObserver: 'readonly',
        ResizeObserver: 'readonly',
        requestAnimationFrame: 'readonly',
        cancelAnimationFrame: 'readonly',
        crypto: 'readonly',
        // Node globals
        process: 'readonly',
        __dirname: 'readonly',
        // Vite globals
        import: 'readonly',
      },
    },
    rules: {
      // Detect unused variables and imports
      'no-unused-vars': ['error', {
        vars: 'all',
        args: 'after-used',
        ignoreRestSiblings: true,
        argsIgnorePattern: '^_',
      }],

      // React specific rules
      'react/jsx-uses-react': 'error',
      'react/jsx-uses-vars': 'error',
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',

      // JSDoc rules - require documentation for functions and classes
      'jsdoc/require-jsdoc': ['error', {
        require: {
          FunctionDeclaration: true,
          MethodDefinition: true,
          ClassDeclaration: true,
          ArrowFunctionExpression: true,
          FunctionExpression: true,
        },
        contexts: [
          'FunctionDeclaration',
          'FunctionExpression',
          'ArrowFunctionExpression',
          'MethodDefinition',
          'ClassDeclaration',
          'VariableDeclaration > VariableDeclarator > ArrowFunctionExpression',
          'VariableDeclaration > VariableDeclarator > FunctionExpression',
        ],
      }],
      'jsdoc/require-description': 'error',
      'jsdoc/require-param': 'error',
      'jsdoc/require-param-description': 'error',
      'jsdoc/require-param-type': 'error',
      'jsdoc/require-returns': 'error',
      'jsdoc/require-returns-description': 'error',
      'jsdoc/require-returns-type': 'error',
      'jsdoc/check-types': 'error',
      'jsdoc/check-param-names': 'error',
      'jsdoc/check-tag-names': 'error',
      // Convert warnings to errors - no cheating!
      'jsdoc/no-undefined-types': 'error',
      'jsdoc/reject-function-type': 'error',
    },
    settings: {
      react: {
        version: 'detect',
      },
    },
  },
];
