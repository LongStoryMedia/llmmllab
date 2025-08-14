#!/bin/bash


function llmmll() {
    kubectl exec -it -n ollama "$(kubectl get pods -n ollama -o jsonpath='{.items[0].metadata.name}')" -- "$@"
}
