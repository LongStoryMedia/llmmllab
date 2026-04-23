

export interface DeleteCertificateResponse {
  /**
   * The ID of the certificate that was deleted.
   */
  id: string;
  /**
   * The object type, must be `certificate.deleted`.
   */
  object: string;
}