'use strict';

// naming convention: https://typescript-eslint.io/rules/naming-convention/
const namingConvention = [
    'error',
    {
        selector: 'default',
        format: ['camelCase'],
    },

    {
        selector: ['import', 'variable', 'function'],
        format: ['camelCase', 'UPPER_CASE', 'PascalCase'],
        leadingUnderscore: 'allow',
        trailingUnderscore: 'allow',
    },

    {
        selector: 'typeLike',
        format: ['PascalCase'],
    },

    {
        selector: ['property', 'parameter'],
        format: ['camelCase', 'PascalCase'],
        leadingUnderscore: 'allow',
    },

    {
        selector: [
            'classProperty',
            'objectLiteralProperty',
            'typeProperty',
            'classMethod',
            'objectLiteralMethod',
            'typeMethod',
            'accessor',
            'enumMember'
        ],
        format: null,
        modifiers: ['requiresQuotes'],
    },
];

module.exports = {
    // Repeated here from eslint-config-xo in case some plugins set something different
    parserOptions: {
        ecmaVersion: 'latest',
        sourceType: 'module',
        ecmaFeatures: {
            jsx: true,
        },
        project: './tsconfig.eslint.json',
        tsconfigRootDir: __dirname,
    },
    parser: '@typescript-eslint/parser',
    extends: [
        'eslint:recommended',
        'plugin:@typescript-eslint/recommended',
        'plugin:react-hooks/recommended',
        'eslint-config-xo',
        'eslint-config-xo-typescript',
        'eslint-config-xo-react'
    ],
    rules: {
        '@typescript-eslint/ban-tslint-comment': 'off',
        '@typescript-eslint/naming-convention': namingConvention,
        'indent': [
            'error',
            2,
            {
                SwitchCase: 1
            }
        ],
        '@typescript-eslint/indent': [
            'error',
            2,
            {
                SwitchCase: 1
            }
        ],
        '@typescript-eslint/consistent-type-definitions': 'off',
        'comma-dangle': 'off',
        '@typescript-eslint/comma-dangle': [
            'error',
            'never'
        ],

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

        'react/jsx-indent': [
            'error',
            2
        ],

        'react/jsx-indent-props': [
            'error',
            2
        ],

        'react/function-component-definition': [
            'error',
            {
                namedComponents: ['function-declaration', 'arrow-function'],
                unnamedComponents: 'arrow-function'
            }
        ],

        'unicorn/prefer-query-selector': 'off',

        'unicorn/prevent-abbreviations': 'off'
    }
};
