import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import pluginReact from 'eslint-plugin-react'
import tseslint from 'typescript-eslint'


// extends: [
//   'eslint:recommended',
//   'plugin:@typescript-eslint/recommended',
//   'plugin:react-hooks/recommended',
//   'eslint-config-xo',
//   'eslint-config-xo-typescript',
//   'eslint-config-xo-react'
// ],
// rules: {
//   "react/jsx-uses-react": "off",
//   "react/react-in-jsx-scope": "off",
//   '@typescript-eslint/ban-tslint-comment': 'off',
//   'indent': [
//     'error',
//     2,
//     {
//       SwitchCase: 1
//     }
//   ], 
//   '@typescript-eslint/indent': [
//     'error',
//     2,
//     {
//       SwitchCase: 1
//     }
//   ],
//   '@typescript-eslint/consistent-type-definitions': 'off',
//   'comma-dangle': 'off',
//   '@typescript-eslint/comma-dangle': [
//     'error',
//     'never'
//   ],

//   "@typescript-eslint/no-unused-vars": [
//     "error",
//     {
//       "args": "all",
//       "argsIgnorePattern": "^_",
//       "caughtErrors": "all",
//       "caughtErrorsIgnorePattern": "^_",
//       "destructuredArrayIgnorePattern": "^_",
//       "varsIgnorePattern": "^_",
//       "ignoreRestSiblings": true
//     }
//   ],

//   'react/jsx-indent': [
//     'error',
//     2
//   ],

//   'react/jsx-indent-props': [
//     'error',
//     2
//   ],

//   'react/function-component-definition': [
//     'error',
//     {
//       namedComponents: ['function-declaration', 'arrow-function'],
//       unnamedComponents: 'arrow-function'
//     }
//   ],

//   'unicorn/prefer-query-selector': 'off',
  
//   'unicorn/prevent-abbreviations': 'off'

export default tseslint.config(
  { ignores: ['dist'] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 'latest',
      globals: globals.browser,
      parserOptions: {
        ecmaVersion: 'latest',
        sourceType: 'module',
        ecmaFeatures: {
          jsx: true,
        }
      }
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
      'react': pluginReact,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      'react-refresh/only-export-components': [
        'warn',
        { allowConstantExport: true },
      ],
        'react/jsx-indent': [
    'error',
    2
  ],

  'react/jsx-indent-props': [
    'error',
    2
  ],
    'indent': [
    'error',
    2,
    {
      SwitchCase: 1
    }
  ],
  'brace-style': [
    'error',
    '1tbs',
    { allowSingleLine: false }
  ],
  'curly': [
    'error',
    'all'
  ],
  '@typescript-eslint/consistent-type-definitions': 'off',
  'comma-dangle': ['error', 'never'],

  "@typescript-eslint/no-unused-vars": [
    "error",
    {
      "args": "all",
      "argsIgnorePattern": "^_",
      "caughtErrors": "all",
      "caughtErrorsIgnorePattern": "^_",
      "destructuredArrayIgnorePattern": "^_",
      "varsIgnorePattern": "^_",
      "ignoreRestSiblings": true
    }
  ],
    }
})
