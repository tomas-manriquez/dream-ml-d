/*
 * Copyright (C) 2025 Leonardo Espinoza Ortiz <leonardo.espinoza.o@usach.cl>
 *
 * This file is part of DREAM ML.
 *
 * DREAM ML is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DREAM ML is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DREAM ML. If not, see <https://www.gnu.org/licenses/>.
 */


import React from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  IconButton,
  Box,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";

const InfoModal = ({ open, onClose, title, content }) => {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: {
          borderRadius: "2px",
          border: "2px solid #00796b",
          fontFamily: "'Roboto Mono', 'Courier New', monospace",
        },
      }}
    >
      <DialogTitle
        sx={{
          backgroundColor: "#004d40",
          color: "#ffffff",
          fontFamily: "'Roboto Mono', 'Courier New', monospace",
          fontWeight: "bold",
          fontSize: "1.1rem",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          borderBottom: "2px solid #00796b",
        }}
      >
        <Box component="span">{title}</Box>
        <IconButton
          onClick={onClose}
          sx={{
            color: "#ffffff",
            "&:hover": { backgroundColor: "rgba(255, 255, 255, 0.1)" },
          }}
        >
          <CloseIcon />
        </IconButton>
      </DialogTitle>

      <DialogContent
        sx={{
          backgroundColor: "#ffffff",
          pt: 3,
          fontFamily: "'Roboto Mono', 'Courier New', monospace",
        }}
      >
        <Typography
          variant="body1"
          sx={{
            fontFamily: "'Roboto Mono', 'Courier New', monospace",
            color: "#004d40",
            lineHeight: 1.8,
            whiteSpace: "pre-line",
          }}
        >
          {content}
        </Typography>
      </DialogContent>

      <DialogActions
        sx={{
          backgroundColor: "#e0f7fa",
          borderTop: "2px solid #00796b",
          padding: "16px 24px",
        }}
      >
        <Button
          onClick={onClose}
          variant="contained"
          sx={{
            backgroundColor: "#00796b",
            color: "#ffffff",
            fontFamily: "'Roboto Mono', 'Courier New', monospace",
            fontWeight: "bold",
            borderRadius: "2px",
            textTransform: "none",
            "&:hover": {
              backgroundColor: "#004d40",
            },
          }}
        >
          Entendido
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default InfoModal;
