# --- Authentication ----------------------------------------------------------

variable "yc_token" {
  description = "IAM token (yc iam create-token). Leave empty if using yc_sa_key_file."
  type        = string
  default     = ""
  sensitive   = true
}

variable "yc_sa_key_file" {
  description = "Path to a service account authorized key JSON. Leave empty if using yc_token."
  type        = string
  default     = ""
}

variable "yc_cloud_id" {
  description = "Yandex Cloud cloud ID."
  type        = string
}

variable "yc_folder_id" {
  description = "Yandex Cloud folder ID."
  type        = string
}

variable "yc_zone" {
  description = "Compute zone."
  type        = string
  default     = "ru-central1-a"
}

# --- VM shape ----------------------------------------------------------------

variable "vm_name" {
  description = "Name of the compute instance."
  type        = string
  default     = "mipt-ml-deploy"
}

variable "vm_cores" {
  description = "Number of vCPU cores."
  type        = number
  default     = 2
}

variable "vm_core_fraction" {
  description = "Guaranteed vCPU performance share (5, 20, 50 or 100)."
  type        = number
  default     = 20
}

variable "vm_memory_gb" {
  description = "RAM in GB."
  type        = number
  default     = 8
}

variable "vm_disk_gb" {
  description = "Boot disk size in GB."
  type        = number
  default     = 30
}

variable "image_family" {
  description = "Image family for the boot disk (Ubuntu LTS)."
  type        = string
  default     = "ubuntu-2204-lts"
}

variable "preemptible" {
  description = "Use a preemptible (cheaper, interruptible) instance."
  type        = bool
  default     = true
}

# --- Existing network (optional) ---------------------------------------------
# If set, no new VPC network/subnet is created (avoids quota limits).

variable "existing_subnet_id" {
  description = "ID of an existing subnet. If set, vpc_network and vpc_subnet resources are skipped."
  type        = string
  default     = ""
}

# --- Access ------------------------------------------------------------------

variable "ssh_user" {
  description = "Login created on the VM."
  type        = string
  default     = "ubuntu"
}

variable "ssh_public_key_path" {
  description = "Path to the SSH public key placed on the VM."
  type        = string
  default     = "~/.ssh/id_rsa.pub"
}
