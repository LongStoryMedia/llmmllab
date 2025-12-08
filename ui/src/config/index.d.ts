import { WebStorageStateStore } from "oidc-client-ts";
declare const _default: {
    server: {
        baseUrl: any;
        apiVersion: string;
    };
    auth: {
        clientId: any;
        clientSecret: any;
        tokenEndpoint: string;
        logoutEndpoint: string;
        usrmgrBaseUrl: string;
        oidc: {
            authority: any;
            client_id: any;
            client_secret: any;
            redirect_uri: string;
            response_type: string;
            scope: string;
            post_logout_redirect_uri: string;
            userStore: WebStorageStateStore;
        };
    };
    theme: {
        light: import("@mui/material").Theme;
        dark: import("@mui/material").Theme;
    };
};
export default _default;
