//! Cyclic redundancy checks — polynomial remainders in `GF(2)[x]` — via the
//! parameterised "Rocksoft" model.
//!
//! A [`Crc`] is fully described by six parameters (`width`, `poly`, `init`,
//! `refin`, `refout`, `xorout`), which between them cover essentially every CRC
//! in practical use. The register is processed most-significant-bit-first with
//! the *normal* (un-reflected) polynomial; input reflection and output
//! reflection are applied around that core, exactly as the model specifies, so
//! the named presets reproduce the published check values bit-for-bit.
//!
//! A streaming [`Digest`] lets a checksum be built incrementally; feeding the
//! same bytes in any chunking yields the same result as [`Crc::checksum`].
//!
//! Integer-only, deterministic, dependency-free — the same charter as the rest
//! of the crate. Widths from 8 to 64 bits are supported.

/// Reverse the low `width` bits of `v`.
fn reflect(v: u64, width: u32) -> u64 {
    let mut r = 0u64;
    for i in 0..width
    {
        if v & (1u64 << i) != 0
        {
            r |= 1u64 << (width - 1 - i);
        }
    }
    r
}

/// A parameterised CRC specification (the Rocksoft model).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Crc {
    width: u32,
    poly: u64,
    init: u64,
    refin: bool,
    refout: bool,
    xorout: u64,
    mask: u64,
}

impl Crc {
    /// Construct a CRC from the six model parameters. `width` must be in
    /// `8..=64`; `poly`, `init` and `xorout` are interpreted as `width`-bit
    /// values (the normal, un-reflected polynomial representation).
    pub const fn new(
        width: u32,
        poly: u64,
        init: u64,
        refin: bool,
        refout: bool,
        xorout: u64,
    ) -> Self {
        assert!(width >= 8 && width <= 64, "CRC width must be in 8..=64");
        let mask = if width == 64
        {
            u64::MAX
        }
        else
        {
            (1u64 << width) - 1
        };
        Crc {
            width,
            poly: poly & mask,
            init: init & mask,
            refin,
            refout,
            xorout: xorout & mask,
            mask,
        }
    }

    /// CRC-32/ISO-HDLC (zlib/PNG/Ethernet): the ubiquitous `0xEDB88320`
    /// reflected CRC-32. Check value `0xCBF43926`.
    pub const fn crc32_iso_hdlc() -> Self {
        Self::new(32, 0x04C1_1DB7, 0xFFFF_FFFF, true, true, 0xFFFF_FFFF)
    }

    /// CRC-32C (Castagnoli), used by iSCSI, SSE4.2, ext4. Check `0xE3069283`.
    pub const fn crc32c() -> Self {
        Self::new(32, 0x1EDC_6F41, 0xFFFF_FFFF, true, true, 0xFFFF_FFFF)
    }

    /// CRC-16/CCITT-FALSE. Check value `0x29B1`.
    pub const fn crc16_ccitt_false() -> Self {
        Self::new(16, 0x1021, 0xFFFF, false, false, 0x0000)
    }

    /// CRC-16/ARC (the "CRC-16" of many tools). Check value `0xBB3D`.
    pub const fn crc16_arc() -> Self {
        Self::new(16, 0x8005, 0x0000, true, true, 0x0000)
    }

    /// CRC-16/XMODEM. Check value `0x31C3`.
    pub const fn crc16_xmodem() -> Self {
        Self::new(16, 0x1021, 0x0000, false, false, 0x0000)
    }

    /// CRC-8/SMBUS (the "CRC-8"). Check value `0xF4`.
    pub const fn crc8_smbus() -> Self {
        Self::new(8, 0x07, 0x00, false, false, 0x00)
    }

    /// CRC-64/XZ (used by the `.xz` container). Check `0x995DC9BBDF1939FA`.
    pub const fn crc64_xz() -> Self {
        Self::new(
            64,
            0x42F0_E1EB_A9EA_3693,
            0xFFFF_FFFF_FFFF_FFFF,
            true,
            true,
            0xFFFF_FFFF_FFFF_FFFF,
        )
    }

    /// The CRC width in bits.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// A fresh streaming digest initialised to this CRC's `init` value.
    pub fn digest(&self) -> Digest<'_> {
        Digest {
            crc: self,
            reg: self.init,
        }
    }

    /// The CRC of a byte slice in one call.
    pub fn checksum(&self, data: &[u8]) -> u64 {
        let mut d = self.digest();
        d.update(data);
        d.finalize()
    }
}

/// An in-progress CRC computation over a stream of bytes.
#[derive(Clone, Debug)]
pub struct Digest<'a> {
    crc: &'a Crc,
    reg: u64,
}

impl Digest<'_> {
    /// Feed more bytes into the running CRC register.
    pub fn update(&mut self, data: &[u8]) {
        let c = self.crc;
        let top = 1u64 << (c.width - 1);
        for &byte in data
        {
            let b = if c.refin
            {
                reflect(byte as u64, 8)
            }
            else
            {
                byte as u64
            };
            self.reg ^= b << (c.width - 8);
            for _ in 0..8
            {
                if self.reg & top != 0
                {
                    self.reg = (self.reg << 1) ^ c.poly;
                }
                else
                {
                    self.reg <<= 1;
                }
                self.reg &= c.mask;
            }
        }
    }

    /// Finish the computation, applying output reflection and the final XOR.
    /// Consumes the digest.
    pub fn finalize(self) -> u64 {
        let c = self.crc;
        let mut r = self.reg;
        if c.refout
        {
            r = reflect(r, c.width);
        }
        (r ^ c.xorout) & c.mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    /// Every preset must reproduce its catalogue "check" value: the CRC of the
    /// ASCII string "123456789".
    #[test]
    fn catalogue_check_values() {
        let msg = b"123456789";
        assert_eq!(Crc::crc32_iso_hdlc().checksum(msg), 0xCBF4_3926);
        assert_eq!(Crc::crc32c().checksum(msg), 0xE306_9283);
        assert_eq!(Crc::crc16_ccitt_false().checksum(msg), 0x29B1);
        assert_eq!(Crc::crc16_arc().checksum(msg), 0xBB3D);
        assert_eq!(Crc::crc16_xmodem().checksum(msg), 0x31C3);
        assert_eq!(Crc::crc8_smbus().checksum(msg), 0xF4);
        assert_eq!(Crc::crc64_xz().checksum(msg), 0x995D_C9BB_DF19_39FA);
    }

    #[test]
    fn empty_and_known_small_inputs() {
        // CRC-32 of the empty string is 0.
        assert_eq!(Crc::crc32_iso_hdlc().checksum(b""), 0);
        // A single byte, reflected model.
        assert_eq!(Crc::crc32_iso_hdlc().checksum(&[0x00]), 0xD202_EF8D);
    }

    #[test]
    fn streaming_matches_one_shot() {
        let presets = [
            Crc::crc32_iso_hdlc(),
            Crc::crc32c(),
            Crc::crc16_ccitt_false(),
            Crc::crc16_arc(),
            Crc::crc8_smbus(),
            Crc::crc64_xz(),
        ];
        let mut s = 0x1234_5678u64;
        for crc in presets
        {
            for _ in 0..50
            {
                let len = (xorshift(&mut s) % 200) as usize;
                let data: Vec<u8> = (0..len).map(|_| (xorshift(&mut s) & 0xff) as u8).collect();
                let one_shot = crc.checksum(&data);
                // feed in random-sized chunks
                let mut d = crc.digest();
                let mut i = 0;
                while i < data.len()
                {
                    let step = 1 + (xorshift(&mut s) % 7) as usize;
                    let end = (i + step).min(data.len());
                    d.update(&data[i..end]);
                    i = end;
                }
                assert_eq!(d.finalize(), one_shot, "chunked != one-shot");
            }
        }
    }

    #[test]
    fn reflect_is_an_involution() {
        let mut s = 0xabcdu64;
        for _ in 0..1000
        {
            for &w in &[8u32, 16, 32, 64]
            {
                let mask = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
                let v = xorshift(&mut s) & mask;
                assert_eq!(reflect(reflect(v, w), w), v);
            }
        }
    }

    #[test]
    fn detects_single_bit_errors() {
        // A CRC must detect every single-bit flip (its Hamming distance ≥ 2).
        let crc = Crc::crc32_iso_hdlc();
        let data = b"the quick brown fox jumps over the lazy dog";
        let good = crc.checksum(data);
        for byte in 0..data.len()
        {
            for bit in 0..8
            {
                let mut corrupt = data.to_vec();
                corrupt[byte] ^= 1 << bit;
                assert_ne!(crc.checksum(&corrupt), good, "missed flip @{byte}:{bit}");
            }
        }
    }
}
