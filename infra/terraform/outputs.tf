output "instance_id" {
  description = "Compute instance ID."
  value       = yandex_compute_instance.this.id
}

output "external_ip" {
  description = "Public IPv4 address of the VM."
  value       = yandex_compute_instance.this.network_interface.0.nat_ip_address
}

output "ssh_command" {
  description = "Ready-to-use SSH command."
  value       = "ssh ${var.ssh_user}@${yandex_compute_instance.this.network_interface.0.nat_ip_address}"
}
