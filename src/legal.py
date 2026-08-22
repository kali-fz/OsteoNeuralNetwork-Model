"""Legal and compliance notices displayed by the local ONNM application.

These notices are a project baseline, not jurisdiction-specific legal advice.
The person or organization operating a deployment should obtain qualified legal
review before using it with real patient data.
"""

TERMS_OF_SERVICE = """
## Terms of Service

**Effective date: 22 August 2026**

### 1. Agreement and operator
OsteoNeuralNetwork-Model ("ONNM", the "Service") is free, open-source research
software operated locally by the person or organization that installed it (the
"Operator"). By creating an account or using the Service, you agree to these Terms,
the Privacy Policy, and the Medical Disclaimer. If you do not agree, do not use it.

### 2. Research-only service
ONNM accepts radiographs and generates experimental classifications, confidence
scores, and visual explanations. It is an unvalidated research prototype, not a
clinical service or medical device. Access may be changed, suspended, or withdrawn
without notice.

### 3. Local research storage
Uploaded images are retained on the Operator's machine in account-isolated folders.
DICOM headers are processed to remove common identifying fields, standard images are
re-encoded to remove metadata, and filenames on disk are replaced with random UUIDs.
Scan metadata remains linked to your local account so history can be shown. Therefore,
data is **de-identified or pseudonymized, not guaranteed anonymous**. Images may be used
locally by the Operator as a research dataset to evaluate and improve model accuracy.
They are never sold by ONNM or transmitted to or shared with third parties by the
software. The software itself performs no cloud upload.

De-identification cannot remove names or identifiers burned into image pixels. You
must inspect and remove visible identifiers before upload and must have all necessary
authority and consent to process the image.

### 4. Account responsibilities
You must provide accurate account information, use a strong unique password, protect
access to the local computer, and promptly stop using an account you believe is
compromised. You may not attempt to access another user's records, bypass safeguards,
introduce malicious files, reverse engineer patient identity, or use the Service for
clinical diagnosis, treatment, emergencies, unlawful surveillance, or discrimination.

### 5. Intellectual property and third-party materials
The ONNM source code is licensed under its repository license. Model checkpoints,
datasets, and dependencies may have separate terms. Uploading content does not transfer
ownership, but you grant the Operator a non-exclusive local license to store, process,
evaluate, and use de-identified copies for the research purposes described above.

### 6. No warranties
To the fullest extent permitted by law, the Service is provided "as is" and "as
available", without warranties of accuracy, fitness for a particular purpose,
non-infringement, availability, security, or regulatory compliance. Model output can
be incomplete, biased, incorrect, or misleading.

### 7. Limitation of liability
To the fullest extent permitted by applicable law, the developers, contributors, and
Operator are not liable for clinical decisions, missed or delayed care, false
positives, false negatives, data loss, loss of profits, or indirect, incidental,
special, consequential, or punitive damages arising from use of the Service. Nothing
in these Terms excludes liability that cannot legally be excluded. An absolute
"zero-liability" waiver may be unenforceable in some jurisdictions.

### 8. Termination and deletion
The Operator may disable access for misuse. Because this is local software, account
and scan deletion must be performed by the Operator against the local database and
storage directory. Deletion from active storage does not automatically erase backups.

### 9. Governing terms and changes
Mandatory law where the Operator and user are located applies. If a provision is
unenforceable, the remainder continues in effect. Updated Terms apply after they are
presented for acceptance; material changes should require renewed consent.

### 10. Contact
Questions, access requests, or deletion requests should be directed to the person or
organization operating this local ONNM installation or to the maintainers through the
repository's published contact channels.
"""

PRIVACY_POLICY = """
## Privacy Policy

**Effective date: 22 August 2026**

### Scope and roles
This Policy covers the local ONNM application. ONNM has no hosted backend. The person
or organization running the installation is the data controller/operator for data
entered into it and is responsible for determining whether HIPAA, UK GDPR, EU GDPR,
state health-privacy law, institutional review, or another regime applies. Installing
this software does **not** itself create compliance with any such regime.

### Data processed
The Service stores: your normalized email address; a salted PBKDF2 password hash
(never your plaintext password); account and Terms-acceptance timestamps; a random
user UUID; uploaded radiograph pixels; the original filename in private scan history;
a random on-disk filename; upload time; model verdict; and confidence score. Streamlit
also keeps transient session state needed to keep you logged in during the browser
session.

### Purposes and legal basis
Data is processed to authenticate users, perform requested local inference, display
private scan history, secure and troubleshoot the installation, and conduct local
research/evaluation intended to improve model accuracy. Depending on jurisdiction,
the Operator must establish an appropriate legal basis and any additional condition
required for health data. Account consent alone may not be sufficient for clinical or
institutional data.

### Storage, de-identification, and security
Credentials and scan metadata are stored in `data/users.db`. Images are stored under
`data/user_uploads/{user_id}/`. Passwords use unique random salts and PBKDF2-HMAC-SHA256.
Database queries are parameterized. User directories and files receive restrictive
permissions where the operating system supports them.

DICOM private tags and common identifying header fields are removed, identifiers are
regenerated, and standard images are re-encoded without metadata. These measures do not
detect identifiers burned into pixels and do not make linked account data anonymous.
The database is not encrypted at rest; full-disk encryption, operating-system access
controls, secure backups, physical security, and malware protection are the Operator's
responsibility.

### Sharing and transfers
The application contains no feature that sells scan data, sends it to advertising
networks, or uploads it to a third-party cloud service. Data stays on the local machine
unless a user or Operator copies it, backs it up, exposes the Streamlit server beyond
loopback, or adds external integrations. Such actions are outside ONNM's built-in flow
and must be separately assessed and disclosed.

### Retention and deletion
Records remain until the Operator deletes the account, database row, image, or storage
directory. Operators should define a documented retention period and secure backup
deletion process appropriate to their research protocol. Users may request access,
correction, export, restriction, or deletion from the local Operator, subject to
applicable law and legitimate research-record obligations.

### Children, incidents, and changes
The Service is not directed to children and should not receive pediatric data without
appropriate authority and safeguards. Suspected unauthorized access should be reported
to the local Operator, who is responsible for investigation and any required notices.
Material Policy changes should be clearly presented and, where required, consented to
again.
"""

MEDICAL_DISCLAIMER = """
## Medical Disclaimer and Liability Notice

ONNM is an **unvalidated research prototype**. It is **not FDA cleared or approved, is
not CE marked, is not MHRA registered, and is not a medical device or clinical
decision-support system**. Its outputs are not a diagnosis, prognosis, treatment
recommendation, or substitute for an examination and interpretation by a qualified
radiologist or treating clinician.

False positives and false negatives are expected. A "normal" result does not exclude
cancer or other disease; a "potential lesion" result does not establish disease.
Confidence scores are model estimates, not probabilities of clinical truth. Grad-CAM
shows model attention and does not prove pathological localization or reasoning.

Do not use ONNM to make, support, defer, or delay patient-care decisions, triage an
emergency, or communicate a diagnosis. Seek qualified medical review for every image
and urgent medical help when clinically indicated.

To the fullest extent permitted by law, the developers, contributors, and local
Operator disclaim responsibility for losses arising from reliance on model output,
including false positives and false negatives. Liability that applicable law does not
permit parties to exclude remains unaffected. This notice is not legal or medical
advice.
"""

COOKIE_NOTICE = """
## Cookie and Session Notice

ONNM does not add advertising, analytics, tracking pixels, or cross-site profiling
cookies. Streamlit may use a technically necessary browser cookie or connection token
to maintain the local session and protect its WebSocket connection. Authentication
state is held in Streamlit server-side session memory and is cleared on logout or when
the session expires; it is not used for advertising.

The Operator should keep the server bound to loopback (`localhost`). If the application
is placed behind a proxy, exposed to a network, or modified to add analytics or an
identity provider, the Operator must update this notice and obtain consent where
required.
"""

