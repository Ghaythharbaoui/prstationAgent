from io import BytesIO
from pathlib import Path

import pandas as pd

from app.ingestion.schemas import LoadedDocument


def load_spreadsheet(
    filename: str,
    content: bytes,
) -> LoadedDocument:
    extension = Path(filename).suffix.lower()

    if not content:
        raise ValueError(
            "The spreadsheet file is empty."
        )

    try:
        if extension == ".csv":
            dataframe = pd.read_csv(
                BytesIO(content),
                sep=None,
                engine="python",
                encoding="utf-8-sig",
            )

            if len(dataframe.columns) == 0:
                raise ValueError(
                    "The CSV file contains no columns."
                )

            extracted_text = dataframe.fillna(
                ""
            ).to_markdown(index=False)

            metadata = {
                "loader": "pandas",
                "extension": extension,
                "row_count": len(dataframe),
                "column_count": len(dataframe.columns),
            }

        elif extension in {".xlsx", ".xlsm"}:
            worksheets = pd.read_excel(
                BytesIO(content),
                sheet_name=None,
                engine="openpyxl",
            )

            sections = []
            worksheet_metadata = []

            for sheet_name, dataframe in worksheets.items():
                table = dataframe.fillna(
                    ""
                ).to_markdown(index=False)

                sections.append(
                    f"## Worksheet: {sheet_name}\n\n{table}"
                )

                worksheet_metadata.append(
                    {
                        "name": sheet_name,
                        "row_count": len(dataframe),
                        "column_count": len(
                            dataframe.columns
                        ),
                    }
                )

            extracted_text = "\n\n".join(
                sections
            ).strip()

            metadata = {
                "loader": "pandas/openpyxl",
                "extension": extension,
                "worksheets": worksheet_metadata,
            }

        else:
            raise ValueError(
                "Unsupported spreadsheet format."
            )

    except ValueError:
        raise

    except Exception as exc:
        raise ValueError(
            "Unable to read the CSV or Excel file."
        ) from exc

    if not extracted_text:
        raise ValueError(
            "No readable tabular data was found."
        )

    return LoadedDocument(
        title=Path(filename).stem or "Untitled",
        content=extracted_text,
        source_type="file",
        source_uri=filename,
        content_format="markdown",
        metadata=metadata,
    )