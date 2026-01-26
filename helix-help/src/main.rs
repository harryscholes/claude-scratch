mod data;

use clap::{Parser, Subcommand};
use colored::Colorize;
use data::{get_commands, get_keybindings, Category, Command, Keybinding, Mode};

#[derive(Parser)]
#[command(name = "helix-help")]
#[command(about = "A CLI tool to help learn Helix editor keybindings and commands", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Search term to find keybindings or commands
    #[arg(global = true)]
    search: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Search keybindings by key, description, or command name
    Key {
        /// The search term
        query: String,
    },
    /// Search typeable commands (used with :)
    Cmd {
        /// The search term
        query: String,
    },
    /// List all keybindings for a specific mode
    Mode {
        /// Mode name: normal, insert, select, view, goto, match, window, space, picker, prompt, unimpaired
        mode: String,
    },
    /// List keybindings by category
    Category {
        /// Category: movement, changes, selection, search, shell, lsp, treesitter, clipboard, navigation, editing, window, buffer
        category: String,
    },
    /// Show all available modes
    Modes,
    /// Show all available categories
    Categories,
    /// Show a quick reference of essential keybindings
    Cheatsheet,
    /// Show differences from Vim
    Vsdiff,
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Key { query }) => search_keybindings(query),
        Some(Commands::Cmd { query }) => search_commands(query),
        Some(Commands::Mode { mode }) => list_mode(mode),
        Some(Commands::Category { category }) => list_category(category),
        Some(Commands::Modes) => list_modes(),
        Some(Commands::Categories) => list_categories(),
        Some(Commands::Cheatsheet) => show_cheatsheet(),
        Some(Commands::Vsdiff) => show_vim_differences(),
        None => {
            if let Some(ref search) = cli.search {
                search_all(search);
            } else {
                show_help();
            }
        }
    }
}

fn show_help() {
    println!("{}", "Helix Editor Help CLI".bold().cyan());
    println!();
    println!("{}",   "USAGE:".bold());
    println!("  helix-help [SEARCH]           Search keybindings and commands");
    println!("  helix-help key <QUERY>        Search keybindings only");
    println!("  helix-help cmd <QUERY>        Search commands only");
    println!("  helix-help mode <MODE>        List keybindings for a mode");
    println!("  helix-help category <CAT>     List keybindings by category");
    println!("  helix-help modes              List all modes");
    println!("  helix-help categories         List all categories");
    println!("  helix-help cheatsheet         Show essential keybindings");
    println!("  helix-help vsdiff             Show differences from Vim");
    println!();
    println!("{}", "EXAMPLES:".bold());
    println!("  helix-help delete             Search for delete-related bindings");
    println!("  helix-help key yank           Search keybindings for 'yank'");
    println!("  helix-help cmd write          Search commands for 'write'");
    println!("  helix-help mode normal        List all normal mode keybindings");
    println!("  helix-help category lsp       List all LSP-related keybindings");
}

fn search_all(query: &str) {
    let query_lower = query.to_lowercase();
    let keybindings = get_keybindings();
    let commands = get_commands();

    let matched_keys: Vec<_> = keybindings
        .iter()
        .filter(|k| {
            k.key.to_lowercase().contains(&query_lower)
                || k.description.to_lowercase().contains(&query_lower)
                || k.command.to_lowercase().contains(&query_lower)
        })
        .collect();

    let matched_cmds: Vec<_> = commands
        .iter()
        .filter(|c| {
            c.name.to_lowercase().contains(&query_lower)
                || c.description.to_lowercase().contains(&query_lower)
                || c.aliases.iter().any(|a| a.to_lowercase().contains(&query_lower))
        })
        .collect();

    if matched_keys.is_empty() && matched_cmds.is_empty() {
        println!("{}", format!("No results found for '{}'", query).yellow());
        return;
    }

    if !matched_keys.is_empty() {
        println!("{}", "KEYBINDINGS".bold().cyan());
        println!("{}", "─".repeat(70));
        print_keybindings(&matched_keys);
    }

    if !matched_cmds.is_empty() {
        if !matched_keys.is_empty() {
            println!();
        }
        println!("{}", "COMMANDS (use with :)".bold().cyan());
        println!("{}", "─".repeat(70));
        print_commands(&matched_cmds);
    }

    println!();
    println!(
        "{}",
        format!(
            "Found {} keybinding(s) and {} command(s)",
            matched_keys.len(),
            matched_cmds.len()
        )
        .dimmed()
    );
}

fn search_keybindings(query: &str) {
    let query_lower = query.to_lowercase();
    let keybindings = get_keybindings();

    let matched: Vec<_> = keybindings
        .iter()
        .filter(|k| {
            k.key.to_lowercase().contains(&query_lower)
                || k.description.to_lowercase().contains(&query_lower)
                || k.command.to_lowercase().contains(&query_lower)
        })
        .collect();

    if matched.is_empty() {
        println!("{}", format!("No keybindings found for '{}'", query).yellow());
        return;
    }

    println!("{}", "KEYBINDINGS".bold().cyan());
    println!("{}", "─".repeat(70));
    print_keybindings(&matched);
    println!();
    println!("{}", format!("Found {} result(s)", matched.len()).dimmed());
}

fn search_commands(query: &str) {
    let query_lower = query.to_lowercase();
    let commands = get_commands();

    let matched: Vec<_> = commands
        .iter()
        .filter(|c| {
            c.name.to_lowercase().contains(&query_lower)
                || c.description.to_lowercase().contains(&query_lower)
                || c.aliases.iter().any(|a| a.to_lowercase().contains(&query_lower))
        })
        .collect();

    if matched.is_empty() {
        println!("{}", format!("No commands found for '{}'", query).yellow());
        return;
    }

    println!("{}", "COMMANDS (use with :)".bold().cyan());
    println!("{}", "─".repeat(70));
    print_commands(&matched);
    println!();
    println!("{}", format!("Found {} result(s)", matched.len()).dimmed());
}

fn list_mode(mode_name: &str) {
    let mode = match mode_name.to_lowercase().as_str() {
        "normal" => Some(Mode::Normal),
        "insert" => Some(Mode::Insert),
        "select" => Some(Mode::Select),
        "view" => Some(Mode::View),
        "goto" | "g" => Some(Mode::Goto),
        "match" | "m" => Some(Mode::Match),
        "window" | "ctrl-w" => Some(Mode::Window),
        "space" => Some(Mode::Space),
        "picker" => Some(Mode::Picker),
        "prompt" => Some(Mode::Prompt),
        "unimpaired" => Some(Mode::Unimpaired),
        _ => None,
    };

    match mode {
        Some(m) => {
            let keybindings = get_keybindings();
            let matched: Vec<_> = keybindings.iter().filter(|k| k.mode == m).collect();

            println!("{} {}", "MODE:".bold(), m.to_string().cyan().bold());
            println!("{}", "─".repeat(70));
            print_keybindings(&matched);
            println!();
            println!("{}", format!("{} keybinding(s)", matched.len()).dimmed());
        }
        None => {
            println!(
                "{}",
                format!("Unknown mode '{}'. Use 'helix-help modes' to see available modes.", mode_name).yellow()
            );
        }
    }
}

fn list_category(cat_name: &str) {
    let category = match cat_name.to_lowercase().as_str() {
        "movement" | "move" => Some(Category::Movement),
        "changes" | "change" => Some(Category::Changes),
        "selection" | "select" => Some(Category::Selection),
        "search" => Some(Category::Search),
        "shell" => Some(Category::Shell),
        "minormodes" | "minor" => Some(Category::MinorModes),
        "lsp" => Some(Category::Lsp),
        "treesitter" | "ts" => Some(Category::TreeSitter),
        "clipboard" | "clip" => Some(Category::Clipboard),
        "navigation" | "nav" => Some(Category::Navigation),
        "editing" | "edit" => Some(Category::Editing),
        "window" | "win" => Some(Category::Window),
        "buffer" | "buf" => Some(Category::Buffer),
        "other" => Some(Category::Other),
        _ => None,
    };

    match category {
        Some(c) => {
            let keybindings = get_keybindings();
            let matched: Vec<_> = keybindings.iter().filter(|k| k.category == c).collect();

            println!("{} {}", "CATEGORY:".bold(), c.to_string().cyan().bold());
            println!("{}", "─".repeat(70));
            print_keybindings(&matched);
            println!();
            println!("{}", format!("{} keybinding(s)", matched.len()).dimmed());
        }
        None => {
            println!(
                "{}",
                format!(
                    "Unknown category '{}'. Use 'helix-help categories' to see available categories.",
                    cat_name
                )
                .yellow()
            );
        }
    }
}

fn list_modes() {
    println!("{}", "AVAILABLE MODES".bold().cyan());
    println!("{}", "─".repeat(50));
    println!("  {}      Default editing mode", "normal".green());
    println!("  {}      Text input mode", "insert".green());
    println!("  {}      Selection extension mode (v)", "select".green());
    println!("  {}        Scrolling/view manipulation (z/Z)", "view".green());
    println!("  {}        Jump to locations (g)", "goto".green());
    println!("  {}       Surround/textobject operations (m)", "match".green());
    println!("  {}      Window/split management (Ctrl-w)", "window".green());
    println!("  {}       Pickers and common actions (Space)", "space".green());
    println!("  {}      Fuzzy finder navigation", "picker".green());
    println!("  {}      Command-line input (:)", "prompt".green());
    println!("  {} Jump between items ([/])", "unimpaired".green());
}

fn list_categories() {
    println!("{}", "AVAILABLE CATEGORIES".bold().cyan());
    println!("{}", "─".repeat(50));
    println!("  {}     Cursor movement", "movement".green());
    println!("  {}      Text modifications", "changes".green());
    println!("  {}    Selection manipulation", "selection".green());
    println!("  {}       Find and replace", "search".green());
    println!("  {}        Shell command integration", "shell".green());
    println!("  {}          Language Server Protocol", "lsp".green());
    println!("  {}   Syntax tree navigation", "treesitter".green());
    println!("  {}    System clipboard operations", "clipboard".green());
    println!("  {}   File and position navigation", "navigation".green());
    println!("  {}      Text editing helpers", "editing".green());
    println!("  {}       Split/window management", "window".green());
    println!("  {}       Buffer operations", "buffer".green());
}

fn show_cheatsheet() {
    println!("{}", "HELIX QUICK REFERENCE".bold().cyan());
    println!("{}", "═".repeat(70));

    println!("\n{}", "Basic Movement".bold());
    println!("  {}        Left/Down/Up/Right", "h j k l".green());
    println!("  {}          Word forward/backward/end", "w b e".green());
    println!("  {}          WORD forward/backward/end", "W B E".green());
    println!("  {}          Find char forward/backward", "f F".green());
    println!("  {}          Find till char forward/backward", "t T".green());
    println!("  {}        Half page down/up", "Ctrl-d/u".green());

    println!("\n{}", "Editing".bold());
    println!("  {}            Insert before/after cursor", "i a".green());
    println!("  {}            Insert at line start/end", "I A".green());
    println!("  {}            Open line below/above", "o O".green());
    println!("  {}            Delete selection", "d".green());
    println!("  {}            Change selection (delete + insert)", "c".green());
    println!("  {}            Yank (copy)", "y".green());
    println!("  {}            Paste after/before", "p P".green());
    println!("  {}            Undo/Redo", "u U".green());
    println!("  {}            Replace character", "r".green());

    println!("\n{}", "Selection (key difference from Vim!)".bold().yellow());
    println!("  {}            Select line (like Vim's V but smarter)", "x".green());
    println!("  {}            Select all in file", "%".green());
    println!("  {}            Select regex matches in selection", "s".green());
    println!("  {}            Split selection on regex", "S".green());
    println!("  {}            Enter extend mode", "v".green());
    println!("  {}            Copy selection to next line (multi-cursor)", "C".green());
    println!("  {}            Collapse selection to single cursor", ";".green());
    println!("  {}       Keep only primary selection", ",".green());

    println!("\n{}", "Goto Mode (g)".bold());
    println!("  {}           Go to file start", "g g".green());
    println!("  {}           Go to file end", "g e".green());
    println!("  {}           Go to line start/end", "g h / g l".green());
    println!("  {}           Go to definition (LSP)", "g d".green());
    println!("  {}           Go to references (LSP)", "g r".green());

    println!("\n{}", "Match Mode (m)".bold());
    println!("  {}           Go to matching bracket", "m m".green());
    println!("  {} Surround with char (e.g., quotes)", "m s <char>".green());
    println!("  {} Select inside/around textobject", "m i / m a".green());

    println!("\n{}", "Space Mode (Space)".bold());
    println!("  {}       File picker", "Space f".green());
    println!("  {}       Buffer picker", "Space b".green());
    println!("  {}       Symbol picker (LSP)", "Space s".green());
    println!("  {}       Show hover docs (LSP)", "Space k".green());
    println!("  {}       Code actions (LSP)", "Space a".green());
    println!("  {}       Rename symbol (LSP)", "Space r".green());
    println!("  {}       Yank to system clipboard", "Space y".green());
    println!("  {}       Paste from system clipboard", "Space p".green());
    println!("  {}       Global search", "Space /".green());

    println!("\n{}", "Window Mode (Ctrl-w)".bold());
    println!("  {}   Vertical/Horizontal split", "Ctrl-w v/s".green());
    println!("  {} Navigate splits", "Ctrl-w h/j/k/l".green());
    println!("  {}   Close window", "Ctrl-w q".green());

    println!("\n{}", "Commands (type : first)".bold());
    println!("  {}           Write file", ":w".green());
    println!("  {}           Quit", ":q".green());
    println!("  {}          Write and quit", ":wq".green());
    println!("  {} Open file", ":o <file>".green());
    println!("  {}       Change theme", ":theme".green());
    println!("  {}       Open config", ":config-open".green());
    println!("  {}       Open tutorial", ":tutor".green());
}

fn show_vim_differences() {
    println!("{}", "HELIX vs VIM: KEY DIFFERENCES".bold().cyan());
    println!("{}", "═".repeat(70));

    println!("\n{}", "1. Selection-First Model".bold().yellow());
    println!("   Helix follows: {} (like Kakoune)", "select → action".cyan());
    println!("   Vim follows:   {} ", "action → motion".dimmed());
    println!("   Example: In Helix, select text first with {}, then {} to delete", "w".green(), "d".green());
    println!("            In Vim, you'd type {} to delete word", "dw".dimmed());

    println!("\n{}", "2. Line Selection".bold().yellow());
    println!("   Helix: {} extends selection line by line (smarter)", "x".green());
    println!("   Vim:   {} enters visual line mode", "V".dimmed());

    println!("\n{}", "3. Multiple Cursors Built-in".bold().yellow());
    println!("   {}  - Copy selection to next line", "C".green());
    println!("   {}  - Select all regex matches in selection", "s".green());
    println!("   {}  - Split selection on regex", "S".green());
    println!("   {}  - Keep only matching selections", "K".green());

    println!("\n{}", "4. No Ex Mode".bold().yellow());
    println!("   Helix uses {} for commands, but it's simpler", ":".green());
    println!("   No complex Ex command language like Vim");

    println!("\n{}", "5. Different Undo".bold().yellow());
    println!("   {} / {} for undo/redo (Vim uses {} / {})", "u".green(), "U".green(), "u".dimmed(), "Ctrl-r".dimmed());
    println!("   {} / {} for earlier/later in history", "Alt-u".green(), "Alt-U".green());

    println!("\n{}", "6. Surround is Built-in".bold().yellow());
    println!("   {} to enter match mode", "m".green());
    println!("   {} to surround with char", "m s <char>".green());
    println!("   {} to delete surrounding", "m d <char>".green());
    println!("   {} to replace surrounding", "m r <from><to>".green());

    println!("\n{}", "7. Text Objects".bold().yellow());
    println!("   {} {} for inside/around (in match mode)", "m i".green(), "m a".green());
    println!("   Objects: {} (word), {} (para), {} (function), {} (class), {} (arg)...",
             "w".cyan(), "p".cyan(), "f".cyan(), "c".cyan(), "a".cyan());

    println!("\n{}", "8. Registers".bold().yellow());
    println!("   {} to select register (same as Vim)", "\"".green());
    println!("   System clipboard via {} / {}", "Space y".green(), "Space p".green());

    println!("\n{}", "9. Macros".bold().yellow());
    println!("   {} to record (Vim uses {})", "Q".green(), "q".dimmed());
    println!("   {} to replay (Vim uses {})", "q".green(), "@".dimmed());

    println!("\n{}", "10. Window Management".bold().yellow());
    println!("   Same {} prefix as Vim", "Ctrl-w".green());
    println!("   But also accessible via {} then {}", "Space".green(), "w".green());

    println!("\n{}", "Common Vim Commands That Work Similarly:".bold().green());
    println!("   {} - movement", "h j k l w b e".cyan());
    println!("   {} - find char", "f F t T".cyan());
    println!("   {} - insert/append", "i a I A o O".cyan());
    println!("   {} - delete, change, yank, paste", "d c y p".cyan());
    println!("   {} - search, next/prev match", "/ n N".cyan());
    println!("   {} - repeat last change", ".".cyan());
}

fn print_keybindings(keybindings: &[&Keybinding]) {
    for k in keybindings {
        println!(
            "  {:<20} {:<35} {}",
            k.key.green(),
            k.description,
            format!("[{}]", k.mode).dimmed()
        );
    }
}

fn print_commands(commands: &[&Command]) {
    for c in commands {
        let aliases = if c.aliases.is_empty() {
            String::new()
        } else {
            format!(" ({})", c.aliases.join(", "))
        };
        println!(
            "  {:<25} {}",
            format!(":{}{}", c.name, aliases).green(),
            c.description
        );
    }
}
