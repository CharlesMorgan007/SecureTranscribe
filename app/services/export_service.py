"""
Export service for generating PDF, CSV, TXT, and JSON exports.
Handles transcript formatting, document generation, and export options.
"""

import logging
import csv
import json
import io
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from app.core.config import get_settings, EXPORT_SETTINGS
from app.models.transcription import Transcription
from app.models.speaker import Speaker
from app.utils.exceptions import ExportError

logger = logging.getLogger(__name__)


class ExportService:
    """
    Export service for generating PDF, CSV, TXT, and JSON exports.
    Handles transcript formatting, document generation, and export options.
    """

    def __init__(self):
        self.settings = get_settings()
        self.supported_formats = EXPORT_SETTINGS["formats"]
        self.include_options = EXPORT_SETTINGS["include_options"]

    def export_transcription(
        self,
        transcription: Transcription,
        export_format: str,
        include_options: Optional[List[str]] = None,
        session=None,
    ) -> bytes:
        """
        Export transcription in specified format.

        Args:
            transcription: Transcription model instance
            export_format: Export format (pdf, csv, txt, json)
            include_options: List of additional content to include
            session: Database session for additional data

        Returns:
            Exported file content as bytes
        """
        try:
            if export_format not in self.supported_formats:
                raise ExportError(f"Unsupported export format: {export_format}")

            include_options = include_options or []

            # Prepare export data
            export_data = self._prepare_export_data(
                transcription, include_options, session
            )

            # Generate export based on format
            if export_format == "pdf":
                return self._export_pdf(export_data)
            elif export_format == "csv":
                return self._export_csv(export_data)
            elif export_format == "txt":
                return self._export_txt(export_data)
            elif export_format == "json":
                return self._export_json(export_data)
            else:
                raise ExportError(f"Unsupported export format: {export_format}")

        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise ExportError(f"Export failed: {str(e)}")

    def _prepare_export_data(
        self,
        transcription: Transcription,
        include_options: List[str],
        session,
    ) -> Dict[str, Any]:
        """Prepare data for export."""
        try:
            # Basic transcription data
            export_data = {
                "transcription": transcription.export_to_dict(),
                "export_timestamp": datetime.utcnow().isoformat(),
                "include_options": include_options,
            }

            # Add speaker information
            if transcription.segments:
                speakers = transcription.get_speaker_list()
                speaker_stats = transcription.get_speaker_stats()
                export_data["speakers"] = speakers
                export_data["speaker_stats"] = speaker_stats

            # Add additional content based on options
            if "meeting_summary" in include_options:
                export_data["meeting_summary"] = self._generate_meeting_summary(
                    transcription
                )

            if "action_items" in include_options:
                export_data["action_items"] = self._extract_action_items(transcription)

            if "next_steps" in include_options:
                export_data["next_steps"] = self._extract_next_steps(transcription)

            if "recommendations" in include_options:
                export_data["recommendations"] = self._generate_recommendations(
                    transcription
                )

            return export_data

        except Exception as e:
            logger.error(f"Failed to prepare export data: {e}")
            raise ExportError(f"Failed to prepare export data: {str(e)}")

    def _export_pdf(self, export_data: Dict[str, Any]) -> bytes:
        """Generate PDF export."""
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)

            # Create styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                "CustomTitle",
                parent=styles["Heading1"],
                fontSize=18,
                spaceAfter=30,
                alignment=1,  # Center
            )
            heading_style = ParagraphStyle(
                "CustomHeading",
                parent=styles["Heading2"],
                fontSize=14,
                spaceAfter=12,
            )
            normal_style = styles["Normal"]

            # Build document content
            story = []

            # Title
            transcription = export_data["transcription"]
            title = f"Transcription: {transcription['original_filename']}"
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 20))

            # Metadata
            metadata_data = [
                ["File Name:", transcription["original_filename"]],
                ["Duration:", transcription["formatted_duration"]],
                ["File Size:", transcription["formatted_file_size"]],
                [
                    "Date Processed:",
                    transcription["created_at"][:10]
                    if transcription["created_at"]
                    else "N/A",
                ],
                ["Number of Speakers:", str(transcription["num_speakers"])],
                ["Language:", transcription.get("language_detected", "Unknown")],
                [
                    "Confidence Score:",
                    f"{transcription.get('confidence_score', 0):.2%}",
                ],
            ]

            metadata_table = Table(metadata_data, colWidths=[2 * inch, 4 * inch])
            metadata_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.lightgrey),
                        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                        ("BACKGROUND", (0, 0), (0, -1), colors.grey),
                        ("TEXTCOLOR", (0, 0), (0, -1), colors.whitesmoke),
                    ]
                )
            )

            story.append(metadata_table)
            story.append(Spacer(1, 20))

            # Meeting Summary
            if "meeting_summary" in export_data:
                story.append(Paragraph("Meeting Summary", heading_style))
                summary = export_data["meeting_summary"]
                story.append(Paragraph(summary, normal_style))
                story.append(Spacer(1, 15))

            # Action Items
            if "action_items" in export_data:
                story.append(Paragraph("Action Items", heading_style))
                action_items = export_data["action_items"]
                for i, item in enumerate(action_items, 1):
                    story.append(Paragraph(f"{i}. {item}", normal_style))
                story.append(Spacer(1, 15))

            # Next Steps
            if "next_steps" in export_data:
                story.append(Paragraph("Next Steps", heading_style))
                next_steps = export_data["next_steps"]
                for i, step in enumerate(next_steps, 1):
                    story.append(Paragraph(f"{i}. {step}", normal_style))
                story.append(Spacer(1, 15))

            # Recommendations
            if "recommendations" in export_data:
                story.append(Paragraph("Recommendations", heading_style))
                recommendations = export_data["recommendations"]
                for i, rec in enumerate(recommendations, 1):
                    story.append(Paragraph(f"{i}. {rec}", normal_style))
                story.append(Spacer(1, 15))

            # Full Transcript
            story.append(Paragraph("Full Transcript", heading_style))

            if transcription.get("segments"):
                for segment in transcription["segments"]:
                    speaker = segment.get("speaker", "Unknown")
                    text = segment.get("text", "")
                    timestamp = self._format_timestamp(segment.get("start_time", 0))

                    # Speaker and timestamp
                    story.append(
                        Paragraph(f"<b>{speaker}</b> [{timestamp}]", normal_style)
                    )
                    # Text
                    story.append(Paragraph(text, normal_style))
                    story.append(Spacer(1, 10))
            else:
                # Full text without segmentation
                story.append(
                    Paragraph(transcription.get("full_transcript", ""), normal_style)
                )

            # Build PDF
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()

        except Exception as e:
            logger.error(f"PDF export failed: {e}")
            raise ExportError(f"PDF export failed: {str(e)}")

    def _export_csv(self, export_data: Dict[str, Any]) -> bytes:
        """Generate CSV export."""
        try:
            buffer = io.StringIO()
            writer = csv.writer(buffer)

            transcription = export_data["transcription"]

            # Write header
            writer.writerow(["Timestamp", "Speaker", "Text", "Confidence", "Duration"])

            # Write segments
            if transcription.get("segments"):
                for segment in transcription["segments"]:
                    timestamp = self._format_timestamp(segment.get("start_time", 0))
                    speaker = segment.get("speaker", "Unknown")
                    text = segment.get("text", "")
                    confidence = f"{segment.get('confidence', 0):.2%}"
                    duration = f"{segment.get('duration', 0):.2f}s"

                    writer.writerow([timestamp, speaker, text, confidence, duration])
            else:
                # Write full transcript as single row
                writer.writerow(
                    [
                        "00:00:00",
                        "Full Transcript",
                        transcription.get("full_transcript", ""),
                        f"{transcription.get('confidence_score', 0):.2%}",
                        transcription.get("file_duration", 0),
                    ]
                )

            # Add metadata rows
            writer.writerow([])
            writer.writerow(["Metadata"])
            writer.writerow(["File Name", transcription["original_filename"]])
            writer.writerow(["Duration", transcription["formatted_duration"]])
            writer.writerow(["Number of Speakers", transcription["num_speakers"]])
            writer.writerow(
                ["Language", transcription.get("language_detected", "Unknown")]
            )
            writer.writerow(["Export Date", export_data["export_timestamp"][:10]])

            # Add additional content
            if "meeting_summary" in export_data:
                writer.writerow([])
                writer.writerow(["Meeting Summary"])
                writer.writerow([export_data["meeting_summary"]])

            if "action_items" in export_data:
                writer.writerow([])
                writer.writerow(["Action Items"])
                for i, item in enumerate(export_data["action_items"], 1):
                    writer.writerow([f"{i}. {item}"])

            buffer.seek(0)
            return buffer.getvalue().encode("utf-8")

        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            raise ExportError(f"CSV export failed: {str(e)}")

    def _export_txt(self, export_data: Dict[str, Any]) -> bytes:
        """Generate TXT export."""
        try:
            buffer = io.StringIO()
            transcription = export_data["transcription"]

            # Header
            buffer.write("=" * 80 + "\n")
            buffer.write(f"TRANSCRIPTION: {transcription['original_filename']}\n")
            buffer.write("=" * 80 + "\n\n")

            # Metadata
            buffer.write("METADATA:\n")
            buffer.write(f"File Name: {transcription['original_filename']}\n")
            buffer.write(f"Duration: {transcription['formatted_duration']}\n")
            buffer.write(f"File Size: {transcription['formatted_file_size']}\n")
            buffer.write(
                f"Date Processed: {transcription['created_at'][:10] if transcription['created_at'] else 'N/A'}\n"
            )
            buffer.write(f"Number of Speakers: {transcription['num_speakers']}\n")
            buffer.write(
                f"Language: {transcription.get('language_detected', 'Unknown')}\n"
            )
            buffer.write(
                f"Confidence Score: {transcription.get('confidence_score', 0):.2%}\n"
            )
            buffer.write("\n")

            # Meeting Summary
            if "meeting_summary" in export_data:
                buffer.write("MEETING SUMMARY:\n")
                buffer.write("-" * 40 + "\n")
                buffer.write(export_data["meeting_summary"] + "\n\n")

            # Action Items
            if "action_items" in export_data:
                buffer.write("ACTION ITEMS:\n")
                buffer.write("-" * 40 + "\n")
                for i, item in enumerate(export_data["action_items"], 1):
                    buffer.write(f"{i}. {item}\n")
                buffer.write("\n")

            # Next Steps
            if "next_steps" in export_data:
                buffer.write("NEXT STEPS:\n")
                buffer.write("-" * 40 + "\n")
                for i, step in enumerate(export_data["next_steps"], 1):
                    buffer.write(f"{i}. {step}\n")
                buffer.write("\n")

            # Recommendations
            if "recommendations" in export_data:
                buffer.write("RECOMMENDATIONS:\n")
                buffer.write("-" * 40 + "\n")
                for i, rec in enumerate(export_data["recommendations"], 1):
                    buffer.write(f"{i}. {rec}\n")
                buffer.write("\n")

            # Full Transcript
            buffer.write("FULL TRANSCRIPT:\n")
            buffer.write("=" * 80 + "\n\n")

            if transcription.get("segments"):
                for segment in transcription["segments"]:
                    speaker = segment.get("speaker", "Unknown")
                    text = segment.get("text", "")
                    timestamp = self._format_timestamp(segment.get("start_time", 0))

                    buffer.write(f"[{timestamp}] {speaker}:\n")
                    buffer.write(f"{text}\n\n")
            else:
                buffer.write(transcription.get("full_transcript", ""))

            # Footer
            buffer.write("\n" + "=" * 80 + "\n")
            buffer.write(f"Exported on: {export_data['export_timestamp']}\n")
            buffer.write("Generated by SecureTranscribe\n")

            buffer.seek(0)
            return buffer.getvalue().encode("utf-8")

        except Exception as e:
            logger.error(f"TXT export failed: {e}")
            raise ExportError(f"TXT export failed: {str(e)}")

    def _export_json(self, export_data: Dict[str, Any]) -> bytes:
        """Generate JSON export."""
        try:
            # Convert to JSON with proper formatting
            json_data = json.dumps(
                export_data,
                indent=EXPORT_SETTINGS["json_indent"],
                ensure_ascii=False,
                default=str,  # Handle datetime objects
            )
            return json_data.encode("utf-8")

        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            raise ExportError(f"JSON export failed: {str(e)}")

    def _generate_meeting_summary(self, transcription: Transcription) -> str:
        """Generate a meeting summary from transcription."""
        try:
            if not transcription.segments:
                return "No segments available for summary generation."

            # Extract key information
            speakers = transcription.get_speaker_list()
            total_duration = transcription.file_duration
            speaker_stats = transcription.get_speaker_stats()

            summary_parts = []

            # Basic information
            summary_parts.append(
                f"This meeting involved {len(speakers)} participants and lasted {transcription.formatted_duration}."
            )

            # Speaker participation
            if speaker_stats:
                summary_parts.append("Key participants:")
                for speaker, stats in speaker_stats.items():
                    participation = (stats["total_duration"] / total_duration) * 100
                    summary_parts.append(
                        f"- {speaker}: {stats['segment_count']} contributions, "
                        f"{participation:.1f}% of meeting time"
                    )

            # Main topics (simplified keyword extraction)
            all_text = " ".join([seg.get("text", "") for seg in transcription.segments])
            topics = self._extract_topics(all_text)
            if topics:
                summary_parts.append(f"Main topics discussed: {', '.join(topics[:5])}")

            return " ".join(summary_parts)

        except Exception as e:
            logger.error(f"Failed to generate meeting summary: {e}")
            return "Unable to generate meeting summary."

    def _extract_action_items(self, transcription: Transcription) -> List[str]:
        """Extract action items from transcription."""
        try:
            action_items = []
            action_keywords = [
                "action",
                "task",
                "follow up",
                "next",
                "will",
                "should",
                "need to",
                "responsible",
                "deadline",
                "complete",
                "finish",
                "assign",
            ]

            if not transcription.segments:
                return action_items

            for segment in transcription.segments:
                text = segment.get("text", "").lower()
                speaker = segment.get("speaker", "Unknown")

                # Simple keyword-based extraction
                for keyword in action_keywords:
                    if keyword in text:
                        # Extract the sentence containing the keyword
                        sentences = text.split(".")
                        for sentence in sentences:
                            if keyword in sentence:
                                action_item = sentence.strip()
                                if action_item and len(action_item) > 10:
                                    action_items.append(f"{speaker}: {action_item}")
                                break

            # Remove duplicates and limit to top 10
            unique_items = list(dict.fromkeys(action_items))
            return unique_items[:10]

        except Exception as e:
            logger.error(f"Failed to extract action items: {e}")
            return []

    def _extract_next_steps(self, transcription: Transcription) -> List[str]:
        """Extract next steps from transcription."""
        try:
            next_steps = []
            next_step_keywords = [
                "next step",
                "moving forward",
                "going forward",
                "future",
                "plan",
                "schedule",
                "meeting",
                "call",
                "discuss",
                "review",
                "implement",
            ]

            if not transcription.segments:
                return next_steps

            for segment in transcription.segments:
                text = segment.get("text", "").lower()
                speaker = segment.get("speaker", "Unknown")

                # Simple keyword-based extraction
                for keyword in next_step_keywords:
                    if keyword in text:
                        # Extract the sentence containing the keyword
                        sentences = text.split(".")
                        for sentence in sentences:
                            if keyword in sentence:
                                next_step = sentence.strip()
                                if next_step and len(next_step) > 10:
                                    next_steps.append(f"{speaker}: {next_step}")
                                break

            # Remove duplicates and limit to top 10
            unique_steps = list(dict.fromkeys(next_steps))
            return unique_steps[:10]

        except Exception as e:
            logger.error(f"Failed to extract next steps: {e}")
            return []

    def _generate_recommendations(self, transcription: Transcription) -> List[str]:
        """Generate recommendations based on transcription analysis."""
        try:
            recommendations = []

            # Analyze participation balance
            speaker_stats = transcription.get_speaker_stats()
            if speaker_stats:
                total_duration = transcription.file_duration
                dominant_speaker = max(
                    speaker_stats.items(), key=lambda x: x[1]["total_duration"]
                )
                dominant_percentage = (
                    dominant_speaker[1]["total_duration"] / total_duration
                ) * 100

                if dominant_percentage > 70:
                    recommendations.append(
                        f"Consider encouraging more balanced participation. {dominant_speaker[0]} "
                        f"spoke {dominant_percentage:.1f}% of the time."
                    )

            # Analyze meeting length
            if transcription.file_duration > 3600:  # Over 1 hour
                recommendations.append(
                    "Consider breaking long meetings into shorter sessions for better focus."
                )

            # Analyze speaker count
            if transcription.num_speakers > 8:
                recommendations.append(
                    "Large meetings with many participants may benefit from a structured agenda "
                    "and designated facilitator."
                )

            # Analyze confidence
            if transcription.confidence_score and transcription.confidence_score < 0.7:
                recommendations.append(
                    "Audio quality could be improved. Consider using better microphones "
                    "or reducing background noise for clearer transcriptions."
                )

            if not recommendations:
                recommendations.append(
                    "Meeting appears to have good structure and participation."
                )

            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return ["Unable to generate recommendations."]

    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text (simplified implementation)."""
        try:
            # Simple keyword extraction - in a real implementation,
            # you might use NLP libraries like spaCy or NLTK
            common_business_topics = [
                "project",
                "budget",
                "timeline",
                "deadline",
                "client",
                "team",
                "strategy",
                "marketing",
                "sales",
                "development",
                "design",
                "testing",
                "deployment",
                "maintenance",
                "support",
                "training",
            ]

            topics = []
            text_lower = text.lower()

            for topic in common_business_topics:
                if topic in text_lower:
                    topics.append(topic.capitalize())

            return topics

        except Exception as e:
            logger.error(f"Failed to extract topics: {e}")
            return []

    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp in HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def get_export_formats(self) -> Dict[str, str]:
        """Get supported export formats."""
        return {
            "pdf": "PDF Document",
            "csv": "CSV Spreadsheet",
            "txt": "Plain Text",
            "json": "JSON Data",
        }

    def get_include_options(self) -> Dict[str, str]:
        """Get available include options."""
        return {
            "meeting_summary": "Meeting Summary",
            "action_items": "Action Items",
            "next_steps": "Next Steps",
            "recommendations": "Recommendations",
        }
