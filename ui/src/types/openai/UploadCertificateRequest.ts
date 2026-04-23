

export interface UploadCertificateRequest {
  /**
   * The certificate content in PEM format
   */
  content: string;
  /**
   * An optional name for the certificate
   */
  name?: string;
}