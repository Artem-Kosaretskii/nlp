values_3bit = [1, 5, 0, 7, 2, 3, 4, 6]
packed_bytes = bytearray(3)

for i, val in enumerate(values_3bit):
    byte_index = (i * 3) // 8
    bit_offset = (i * 3) % 8

    val = val % 8

    shifted = val << bit_offset
    packed_bytes[byte_index] |= shifted % 256

    if bit_offset > 5:
        remaining = val >> (8 - bit_offset)
        packed_bytes[byte_index + 1] |= remaining % 256

print(f"Default: {values_3bit}")
print(f"Packed: {[b for b in packed_bytes]}")

unpacked_values = []
for i in range(8):
    byte_index = (i * 3) // 8
    bit_offset = (i * 3) % 8

    value = (packed_bytes[byte_index] >> bit_offset) % 8

    if bit_offset > 5:
        bits_from_next = 8 - bit_offset
        next_part = (packed_bytes[byte_index + 1] << bits_from_next) % 256
        value = (value + next_part) % 8

    unpacked_values.append(value)

print(f"Unpacked: {unpacked_values}")
print(f"Matching: {values_3bit == unpacked_values}")
