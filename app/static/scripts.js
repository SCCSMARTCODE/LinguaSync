document.getElementById("translation-form").addEventListener("submit", async (event) => {
    event.preventDefault();

    const englishText = document.getElementById("english-input").value;
    const frenchOutput = document.getElementById("french-output");

    // Clear the output field
    frenchOutput.textContent = "Translating...";

    try {
        const response = await fetch("/translate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: englishText }),
        });

        if (response.ok) {
            const data = await response.json();
            frenchOutput.textContent = data.translation;
        } else {
            frenchOutput.textContent = "Translation failed. Please try again.";
        }
    } catch (error) {
        frenchOutput.textContent = "An error occurred. Please try again.";
    }
});
